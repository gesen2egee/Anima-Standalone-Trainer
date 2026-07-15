"""Training helpers for Self-Flow network fine-tuning.

The paper uses an EMA copy of the complete generative model.  Network training
keeps the base model frozen, so the equivalent parameter-efficient teacher is
an EMA copy of the DiT adapter parameters on top of the same frozen base.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTOR_MODULE_NAME = "_self_flow_projector"


class ProjectionHead(nn.Module):
    """Lightweight per-token MLP used only by the student representation loss."""

    def __init__(self, feature_dim: int, projection_dim: int):
        super().__init__()
        projection_dim = min(feature_dim, projection_dim)
        self.in_proj = nn.Linear(feature_dim, projection_dim)
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(projection_dim, feature_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(hidden_states)))


def representation_loss(
    projector: nn.Module,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """Return the cosine Self-Flow loss averaged over batch and image tokens."""

    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "Self-Flow student and teacher features must have the same shape, "
            f"got {tuple(student_hidden.shape)} and {tuple(teacher_hidden.shape)}"
        )
    projected = projector(student_hidden)
    cosine = F.cosine_similarity(projected.float(), teacher_hidden.detach().float(), dim=-1)
    return (1.0 - cosine).mean()


def sample_dual_timestep_mask(
    batch_size: int,
    token_height: int,
    token_width: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample the Bernoulli token mask from Self-Flow Eq. 4."""

    return torch.rand(batch_size, 1, token_height, token_width, device=device) < ratio


def mix_token_timesteps(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mix two per-sample timesteps into a token grid."""

    first = first.reshape(first.shape[0], 1, 1, 1)
    second = second.reshape(second.shape[0], 1, 1, 1)
    return torch.where(mask, second, first)


def expand_token_grid(token_grid: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Expand a patch-token grid to the latent pixel grid used for noising."""

    if patch_size == 1:
        return token_grid
    return token_grid.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)


class AdapterEMA:
    """EMA state and safe parameter swapping for a parameter-efficient teacher."""

    def __init__(self, named_parameters: Mapping[str, nn.Parameter], decay: float):
        self.decay = float(decay)
        self.num_updates = 0
        self._parameters: Dict[str, nn.Parameter] = dict(named_parameters)
        if not self._parameters:
            raise ValueError("Self-Flow could not find trainable DiT adapter parameters")
        self.shadow: Dict[str, torch.Tensor] = {
            name: parameter.detach().clone() for name, parameter in self._parameters.items()
        }

    def _sync_shadow(self) -> None:
        for name, parameter in self._parameters.items():
            shadow = self.shadow[name]
            if shadow.device != parameter.device or shadow.dtype != parameter.dtype:
                self.shadow[name] = shadow.to(device=parameter.device, dtype=parameter.dtype)

    @torch.no_grad()
    def update(self) -> None:
        self._sync_shadow()
        one_minus_decay = 1.0 - self.decay
        for name, parameter in self._parameters.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=one_minus_decay)
        self.num_updates += 1

    @contextmanager
    def apply(self) -> Iterator[None]:
        """Temporarily replace live adapter parameters with their EMA values."""

        self._sync_shadow()
        backup = {name: parameter.detach().clone() for name, parameter in self._parameters.items()}
        try:
            with torch.no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict.get("num_updates", 0))
        loaded = state_dict["shadow"]
        missing = set(self._parameters) - set(loaded)
        unexpected = set(loaded) - set(self._parameters)
        if missing or unexpected:
            raise ValueError(
                f"Self-Flow EMA parameter mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.shadow = {name: value.detach().clone() for name, value in loaded.items()}


def dit_adapter_named_parameters(network: nn.Module) -> Dict[str, nn.Parameter]:
    """Collect DiT-only adapter parameters across LoRA and LyCORIS-style modules."""

    unet_adapters: Iterable[nn.Module] = getattr(network, "unet_loras", ())
    parameter_ids = {
        id(parameter)
        for adapter in unet_adapters
        for parameter in adapter.parameters()
        if parameter.requires_grad
    }
    return {
        name: parameter
        for name, parameter in network.named_parameters()
        if id(parameter) in parameter_ids and not name.startswith(PROJECTOR_MODULE_NAME + ".")
    }


@contextmanager
def adapter_output_context(network: nn.Module, ema: AdapterEMA | None, save_ema: bool) -> Iterator[None]:
    """Hide the training-only projector and optionally expose EMA adapter weights."""

    projector = network._modules.pop(PROJECTOR_MODULE_NAME, None)
    try:
        if save_ema and ema is not None:
            with ema.apply():
                yield
        else:
            yield
    finally:
        if projector is not None:
            network.add_module(PROJECTOR_MODULE_NAME, projector)
