"""Krea 2 LoRA adapter for the shared kohya-style training pipeline."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from networks import lora_flux


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a LoRA network that targets every Linear in the Krea2 DiT.

    The Krea2 text encoder is frozen and its output is cached, so only the DiT is
    wrapped. ``lora_flux.LoRANetwork`` already provides the generic Linear walker,
    optimizer parameter groups, save format, and merge behavior used by Anima.
    """
    return lora_flux.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        [None],
        unet,
        neuron_dropout=neuron_dropout,
        train_blocks="all",
        target_all_linears=True,
        **kwargs,
    )


def create_network_from_weights(
    multiplier: float,
    file: str,
    vae: Optional[nn.Module],
    text_encoders: Optional[List[nn.Module]],
    unet: Optional[nn.Module],
    weights_sd: Optional[Dict[str, torch.Tensor]] = None,
    for_inference: bool = False,
    **kwargs,
):
    return lora_flux.create_network_from_weights(
        multiplier, file, vae, [None], unet, weights_sd, for_inference, target_all_linears=True, **kwargs
    )
