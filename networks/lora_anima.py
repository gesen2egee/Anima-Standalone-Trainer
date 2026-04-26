# LoRA network module for Anima 
import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from networks.lora_flux import LoRAModule, LoRAInfModule


# ---------------------------------------------------------------------------
#  TP-aware LoRA modules
#  These subclasses handle the extra communication that Tensor Parallel and
#  Sequence Parallel layers inject into the forward path.  The base LoRAModule
#  only sees the *post-communication* output from org_forward, but the LoRA
#  path (lora_down → lora_up) operates on the *pre-communication* input `x`.
#  Without correction the shapes or semantics would mismatch.
# ---------------------------------------------------------------------------

# Lazy-import flag — set True once wd_parallel is confirmed importable.
_WDP_AVAILABLE: Optional[bool] = None


def _try_import_wdp():
    global _WDP_AVAILABLE
    if _WDP_AVAILABLE is not None:
        return _WDP_AVAILABLE
    try:
        import wd_parallel  # noqa: F401
        _WDP_AVAILABLE = True
    except ImportError:
        _WDP_AVAILABLE = False
    return _WDP_AVAILABLE


def _is_tp_linear(module: torch.nn.Module) -> bool:
    """Return True if *module* is a wd_parallel Column/RowParallelLinear."""
    name = module.__class__.__name__
    return name in ("ColumnParallelLinear", "RowParallelLinear")


class ColumnParallelLoRAModule(LoRAModule):
    """LoRA adapter for ColumnParallelLinear.

    Column-parallel forward:
      TP-only:  copy-to-TP (identity fwd) → F.linear
      SP:       all-gather along seq_dim   → F.linear

    The LoRA path must apply the same input transform so that lora_down
    sees the same tensor shape that the base weight sees.
    For TP-only the input is already replicated — nothing to do.
    For SP the input is (…, S/tp, …) and we must all-gather before lora_down.
    """

    def __init__(self, *args, tp_group=None, seq_dim=1, use_sp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # save before apply_to() deletes it
        self._tp_group = tp_group
        self._seq_dim = seq_dim
        self._use_sp = use_sp

    def apply_to(self):
        org_module = self.org_module
        super().apply_to()
        org_module._tp_lora_adapter = self

    def _prepare_tp_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_group is None or self._tp_group.size() <= 1:
            return x
        if self._use_sp:
            from wd_parallel import gather_from_sp_region
            return gather_from_sp_region(x, self._tp_group, self._seq_dim)
        if getattr(self.org_module_ref[0], "skip_input_grad", False):
            from wd_parallel import copy_to_tp_region_no_input_grad
            return copy_to_tp_region_no_input_grad(x, self._tp_group)
        from wd_parallel import copy_to_tp_region
        return copy_to_tp_region(x, self._tp_group)

    def _apply_lora_path(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        lx = self.lora_down(x)
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale
        return self.lora_up(lx), scale

    def forward_from_prepared_input(self, shared_x: torch.Tensor) -> torch.Tensor:
        org_module = self.org_module_ref[0]
        org_forwarded = torch.nn.functional.linear(shared_x, org_module.weight, org_module.bias)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx, scale = self._apply_lora_path(shared_x)
        output = org_forwarded + lx * self.multiplier * scale

        if self.training and not getattr(ColumnParallelLoRAModule, '_hooks_registered', False):
            ColumnParallelLoRAModule._hooks_registered = True
            name = self.lora_name

            def _hook_output(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG ColPar] A(output grad): {status}  {_n}  nan={torch.isnan(grad).sum().item()}  shape={tuple(grad.shape)}")
                return grad
            if output.requires_grad:
                output.register_hook(_hook_output)

            def _hook_lx(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG ColPar] B(lx grad, after lora_up): {status}  nan={torch.isnan(grad).sum().item()}")
                return grad
            if lx.requires_grad:
                lx.register_hook(_hook_lx)

            def _hook_shared_x(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG ColPar] C(shared_x grad, after TP/SP input bwd): {status}  nan={torch.isnan(grad).sum().item()}")
                return grad
            if shared_x.requires_grad:
                shared_x.register_hook(_hook_shared_x)

        return output

    def forward(self, x):
        shared_x = self._prepare_tp_input(x)
        return self.forward_from_prepared_input(shared_x)


class PackedColumnParallelLoRAModule(LoRAModule):
    """LoRA adapter for packed ColumnParallelLinear (qkv_proj / kv_proj)."""

    def __init__(self, *args, tp_group=None, seq_dim=1, use_sp=False, logical_part_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]
        self._tp_group = tp_group
        self._seq_dim = seq_dim
        self._use_sp = use_sp
        self._packed_parts = int(getattr(self.org_module, "packed_parts", 0) or 0)
        if self._packed_parts < 2:
            raise ValueError("PackedColumnParallelLoRAModule requires org_module.packed_parts >= 2")

        local_part = int(getattr(self.org_module, "local_part_size", self.org_module.out_features // self._packed_parts))
        in_dim = self.org_module.in_features
        if logical_part_names is None:
            if self.lora_name.endswith("_qkv_proj"):
                logical_part_names = ["q_proj", "k_proj", "v_proj"]
            elif self.lora_name.endswith("_kv_proj"):
                logical_part_names = ["k_proj", "v_proj"]
            else:
                logical_part_names = [f"part{i}" for i in range(self._packed_parts)]
        self.logical_part_names = list(logical_part_names)

        self.lora_down = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(self._packed_parts)]
        )
        self.lora_up = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, local_part, bias=False) for _ in range(self._packed_parts)]
        )
        for down in self.lora_down:
            torch.nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        for up in self.lora_up:
            torch.nn.init.zeros_(up.weight)

    def apply_to(self):
        org_module = self.org_module
        super().apply_to()
        org_module._tp_lora_adapter = self

    def _prepare_tp_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_group is None or self._tp_group.size() <= 1:
            return x
        if self._use_sp:
            from wd_parallel import gather_from_sp_region
            return gather_from_sp_region(x, self._tp_group, self._seq_dim)
        if getattr(self.org_module_ref[0], "skip_input_grad", False):
            from wd_parallel import copy_to_tp_region_no_input_grad
            return copy_to_tp_region_no_input_grad(x, self._tp_group)
        from wd_parallel import copy_to_tp_region
        return copy_to_tp_region(x, self._tp_group)

    def forward_from_prepared_input(self, shared_x: torch.Tensor) -> torch.Tensor:
        org_module = self.org_module_ref[0]
        org_forwarded = torch.nn.functional.linear(shared_x, org_module.weight, org_module.bias)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        # Execute packed LoRA as a single down projection plus a single batched
        # up projection while preserving the existing per-part parameter layout.
        down_weight = torch.cat([down.weight for down in self.lora_down], dim=0)
        lx = torch.nn.functional.linear(shared_x, down_weight)
        lx = lx.view(*shared_x.shape[:-1], self._packed_parts, self.lora_dim)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self._packed_parts, self.lora_dim), device=lx.device) > self.rank_dropout
            if lx.dim() == 4:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        up_weight = torch.stack([up.weight for up in self.lora_up], dim=0)
        outs = torch.einsum("...pr,por->...po", lx, up_weight)
        outs = outs.reshape(*shared_x.shape[:-1], -1)

        return org_forwarded + outs * (self.multiplier * scale)

    def forward(self, x):
        shared_x = self._prepare_tp_input(x)
        return self.forward_from_prepared_input(shared_x)


class RowParallelLoRAModule(LoRAModule):
    """LoRA adapter for RowParallelLinear.

    Row-parallel forward:
      TP-only:  F.linear → all-reduce
      SP:       F.linear → reduce-scatter along seq_dim

    The LoRA path operates on the same sharded input and must apply the
    same output reduction so its contribution matches org_forwarded.
    """

    def __init__(self, *args, tp_group=None, seq_dim=1, use_sp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # save before apply_to() deletes it
        self._tp_group = tp_group
        self._seq_dim = seq_dim
        self._use_sp = use_sp

    def _reduce_tp_output(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_group is None or self._tp_group.size() <= 1:
            return x
        if self._use_sp:
            from wd_parallel import reduce_scatter_to_sp_region
            return reduce_scatter_to_sp_region(x, self._tp_group, self._seq_dim)
        import torch.distributed as dist
        x = x.contiguous()
        dist.all_reduce(x, group=self._tp_group)
        return x

    def forward(self, x):
        org_module = self.org_module_ref[0]
        x_local = x
        if x_local.size(-1) < org_module.in_features:
            x_local = torch.nn.functional.pad(x_local, (0, org_module.in_features - x_local.size(-1)))
        org_local = torch.nn.functional.linear(x_local, org_module.weight, None)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                output = self._reduce_tp_output(org_local)
                if org_module.bias is not None:
                    output = output + org_module.bias
                return output

        lx = self.lora_down(x_local)

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            lx = lx * mask
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx) * self.multiplier * scale
        combined_local = org_local + lx
        lx_pre_scatter = combined_local  # save ref for hook before scatter changes the node
        output = self._reduce_tp_output(combined_local)
        if org_module.bias is not None:
            output = output + org_module.bias

        # Only register hooks for the very first RowPar module.
        # Hooks fire in backward order: A (output) → B (lx post-scatter) → C (lx pre-scatter).
        # Reading the output:
        #   A NaN only        → upstream (base model backward) is already NaN
        #   A+B NaN, C finite → reduce_scatter backward (all_gather) introduces NaN
        #   A+B+C NaN         → NaN from lora_up backward or upstream
        if self.training and not getattr(RowParallelLoRAModule, '_hooks_registered', False):
            RowParallelLoRAModule._hooks_registered = True
            name = self.lora_name

            def _hook_output(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG RowPar] A(output grad): {status}  {_n}  nan={torch.isnan(grad).sum().item()}  shape={tuple(grad.shape)}")
                return grad
            if output.requires_grad:
                output.register_hook(_hook_output)

            def _hook_lx_post(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG RowPar] B(combined post-scatter grad): {status}  nan={torch.isnan(grad).sum().item()}")
                return grad
            if lx.requires_grad:
                lx.register_hook(_hook_lx_post)

            def _hook_lx_pre(grad, _n=name):
                from tqdm import tqdm
                status = "NaN" if not torch.isfinite(grad).all() else "ok"
                tqdm.write(f"[NaN DIAG RowPar] C(combined pre-scatter grad): {status}  nan={torch.isnan(grad).sum().item()}")
                return grad
            if lx_pre_scatter.requires_grad:
                lx_pre_scatter.register_hook(_hook_lx_pre)

        return output


def _select_lora_class(child_module, tp_group=None, use_sp=False, seq_dim=1):
    """Pick the right LoRAModule class based on the target layer type.

    Returns (module_class, extra_kwargs) where extra_kwargs are passed
    to the LoRAModule constructor in addition to the standard args.
    """
    cls_name = child_module.__class__.__name__
    if cls_name == "ColumnParallelLinear":
        packed_parts = int(getattr(child_module, "packed_parts", 0) or 0)
        if packed_parts:
            return PackedColumnParallelLoRAModule, dict(
                tp_group=tp_group,
                seq_dim=seq_dim,
                use_sp=getattr(child_module, 'sequence_parallel', use_sp),
                logical_part_names=None,
            )
        return ColumnParallelLoRAModule, dict(tp_group=tp_group, seq_dim=seq_dim,
                                               use_sp=getattr(child_module, 'sequence_parallel', use_sp))
    elif cls_name == "RowParallelLinear":
        return RowParallelLoRAModule, dict(tp_group=tp_group, seq_dim=seq_dim,
                                            use_sp=getattr(child_module, 'sequence_parallel', use_sp))
    else:
        return None, {}  # use the default module_class


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    # type_dims: [self_attn_dim, cross_attn_dim, mlp_dim, mod_dim, llm_adapter_dim]
    self_attn_dim = kwargs.get("self_attn_dim", None)
    cross_attn_dim = kwargs.get("cross_attn_dim", None)
    mlp_dim = kwargs.get("mlp_dim", None)
    mod_dim = kwargs.get("mod_dim", None)
    llm_adapter_dim = kwargs.get("llm_adapter_dim", None)

    if self_attn_dim is not None:
        self_attn_dim = int(self_attn_dim)
    if cross_attn_dim is not None:
        cross_attn_dim = int(cross_attn_dim)
    if mlp_dim is not None:
        mlp_dim = int(mlp_dim)
    if mod_dim is not None:
        mod_dim = int(mod_dim)
    if llm_adapter_dim is not None:
        llm_adapter_dim = int(llm_adapter_dim)

    type_dims = [self_attn_dim, cross_attn_dim, mlp_dim, mod_dim, llm_adapter_dim]
    if all([d is None for d in type_dims]):
        type_dims = None

    # emb_dims: [x_embedder, t_embedder, final_layer]
    emb_dims = kwargs.get("emb_dims", None)
    if emb_dims is not None:
        emb_dims = emb_dims.strip()
        if emb_dims.startswith("[") and emb_dims.endswith("]"):
            emb_dims = emb_dims[1:-1]
        emb_dims = [int(d) for d in emb_dims.split(",")]
        assert len(emb_dims) == 3, f"invalid emb_dims: {emb_dims}, must be 3 dimensions (x_embedder, t_embedder, final_layer)"

    # block selection
    def parse_block_selection(selection: str, total_blocks: int) -> List[bool]:
        if selection == "all":
            return [True] * total_blocks
        if selection == "none" or selection == "":
            return [False] * total_blocks

        selected = [False] * total_blocks
        ranges = selection.split(",")
        for r in ranges:
            if "-" in r:
                start, end = map(str.strip, r.split("-"))
                start, end = int(start), int(end)
                assert 0 <= start < total_blocks and 0 <= end < total_blocks and start <= end
                for i in range(start, end + 1):
                    selected[i] = True
            else:
                index = int(r)
                assert 0 <= index < total_blocks
                selected[index] = True
        return selected

    train_block_indices = kwargs.get("train_block_indices", None)
    if train_block_indices is not None:
        num_blocks = len(unet.blocks) if hasattr(unet, 'blocks') else 999
        train_block_indices = parse_block_selection(train_block_indices, num_blocks)

    # train LLM adapter
    train_llm_adapter = kwargs.get("train_llm_adapter", False)
    if train_llm_adapter is not None:
        train_llm_adapter = True if train_llm_adapter == "True" else False

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        type_dims=type_dims,
        emb_dims=emb_dims,
        train_block_indices=train_block_indices,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    # TP/SP fused-QKV training exposes qkv_proj/kv_proj internally, but saved
    # LoRAs keep standard q_proj/k_proj/v_proj names.  Mirror those dimensions
    # onto the packed internal names so dim-from-weights works after fusion.
    for lora_name, dim in list(modules_dim.items()):
        packed_name = None
        if lora_name.endswith("_self_attn_q_proj") or lora_name.endswith("_self_attn_k_proj") or lora_name.endswith("_self_attn_v_proj"):
            packed_name = lora_name.rsplit("_", 2)[0] + "_qkv_proj"
        elif lora_name.endswith("_cross_attn_k_proj") or lora_name.endswith("_cross_attn_v_proj"):
            packed_name = lora_name.rsplit("_", 2)[0] + "_kv_proj"
        if packed_name is not None:
            modules_dim.setdefault(packed_name, dim)
            if lora_name in modules_alpha:
                modules_alpha.setdefault(packed_name, modules_alpha[lora_name])

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks
    ANIMA_TARGET_REPLACE_MODULE = ["Block"]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te1"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_llm_adapter: bool = False,
        type_dims: Optional[List[int]] = None,
        emb_dims: Optional[List[int]] = None,
        train_block_indices: Optional[List[bool]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_llm_adapter = train_llm_adapter
        self.type_dims = type_dims
        self.emb_dims = emb_dims
        self.train_block_indices = train_block_indices

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
            if self.emb_dims is None:
                self.emb_dims = [0] * 3
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
            include_conv2d_if_filter: bool = False,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = (
                self.LORA_PREFIX_ANIMA
                if is_unet
                else self.LORA_PREFIX_TEXT_ENCODER
            )

            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        _cls_name = child_module.__class__.__name__
                        is_linear = _cls_name in ("Linear", "ColumnParallelLinear", "RowParallelLinear")
                        is_conv2d = _cls_name == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + (name + "." if name else "") + child_name
                            lora_name = lora_name.replace(".", "_")

                            force_incl_conv2d = False
                            if filter is not None:
                                if filter not in lora_name:
                                    continue
                                force_incl_conv2d = include_conv2d_if_filter

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha_val = self.alpha

                                    if is_unet and type_dims is not None:
                                        # type_dims = [self_attn_dim, cross_attn_dim, mlp_dim, mod_dim, llm_adapter_dim]
                                        # Order matters: check most specific identifiers first to avoid mismatches.
                                        identifier_order = [
                                            (4, ("llm_adapter",)),         
                                            (3, ("adaln_modulation",)),   
                                            (0, ("self_attn",)),
                                            (1, ("cross_attn",)),
                                            (2, ("mlp",)),
                                        ]
                                        for idx, ids in identifier_order:
                                            d = type_dims[idx]
                                            if d is not None and all(id_str in lora_name for id_str in ids):
                                                dim = d  # 0 means skip
                                                break

                                    # block index filtering
                                    if is_unet and dim and self.train_block_indices is not None and "blocks_" in lora_name:
                                        # Extract block index from lora_name: "lora_unet_blocks_0_self_attn..."
                                        parts = lora_name.split("_")
                                        for pi, part in enumerate(parts):
                                            if part == "blocks" and pi + 1 < len(parts):
                                                try:
                                                    block_index = int(parts[pi + 1])
                                                    if not self.train_block_indices[block_index]:
                                                        dim = 0
                                                except (ValueError, IndexError):
                                                    pass
                                                break

                                elif force_incl_conv2d:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha_val = self.alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            # Check if this is a TP parallel layer — use TP-aware LoRA class
                            tp_cls, tp_kwargs = _select_lora_class(
                                child_module,
                                tp_group=getattr(root_module, '_wdp_groups', None) and getattr(root_module, '_wdp_groups').tp,
                                use_sp=False,  # read from child_module attribute
                                seq_dim=1,     # Anima uses batch-first (B, S, D)
                            )
                            actual_class = tp_cls if tp_cls is not None else module_class

                            lora = actual_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha_val,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                **tp_kwargs,
                            )
                            loras.append(lora)

                    if target_replace_modules is None:
                        break
            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i+1}:")
                te_loras, te_skipped = create_modules(
                    False, i, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                )
                logger.info(f"create LoRA for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        # emb_dims: [x_embedder, t_embedder, final_layer]
        if self.emb_dims:
            for filter_name, in_dim in zip(
                ["x_embedder", "t_embedder", "final_layer"],
                self.emb_dims,
            ):
                loras, _ = create_modules(
                    True, None, unet, None,
                    filter=filter_name, default_dim=in_dim,
                    include_conv2d_if_filter=(filter_name == "x_embedder"),
                )
                self.unet_loras.extend(loras)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        # assertion: no duplicate names
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    @staticmethod
    def _packed_lora_standard_name(lora_name: str, logical_name: str) -> str:
        if lora_name.endswith("_qkv_proj"):
            return lora_name[: -len("_qkv_proj")] + f"_{logical_name}"
        if lora_name.endswith("_kv_proj"):
            return lora_name[: -len("_kv_proj")] + f"_{logical_name}"
        return f"{lora_name}_{logical_name}"

    def _state_dict_to_standard_packed_lora_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        converted = dict(state_dict)
        for lora in self.text_encoder_loras + self.unet_loras:
            if not isinstance(lora, PackedColumnParallelLoRAModule):
                continue
            prefix = lora.lora_name
            alpha_key = f"{prefix}.alpha"
            alpha = converted.get(alpha_key, lora.alpha.detach())
            for idx, logical_name in enumerate(lora.logical_part_names):
                std_prefix = self._packed_lora_standard_name(prefix, logical_name)
                down_key = f"{prefix}.lora_down.{idx}.weight"
                up_key = f"{prefix}.lora_up.{idx}.weight"
                if down_key in converted:
                    converted[f"{std_prefix}.lora_down.weight"] = converted[down_key]
                if up_key in converted:
                    converted[f"{std_prefix}.lora_up.weight"] = converted[up_key]
                converted[f"{std_prefix}.alpha"] = alpha
            for key in list(converted.keys()):
                if key == alpha_key or key.startswith(f"{prefix}.lora_down.") or key.startswith(f"{prefix}.lora_up."):
                    del converted[key]
        return converted

    def _state_dict_from_standard_packed_lora_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        converted = dict(state_dict)
        for lora in self.text_encoder_loras + self.unet_loras:
            if not isinstance(lora, PackedColumnParallelLoRAModule):
                continue
            prefix = lora.lora_name
            found_alpha = None
            for idx, logical_name in enumerate(lora.logical_part_names):
                std_prefix = self._packed_lora_standard_name(prefix, logical_name)
                std_down = f"{std_prefix}.lora_down.weight"
                std_up = f"{std_prefix}.lora_up.weight"
                std_alpha = f"{std_prefix}.alpha"
                if std_down in converted:
                    converted[f"{prefix}.lora_down.{idx}.weight"] = converted[std_down]
                    del converted[std_down]
                if std_up in converted:
                    converted[f"{prefix}.lora_up.{idx}.weight"] = converted[std_up]
                    del converted[std_up]
                if std_alpha in converted:
                    found_alpha = converted[std_alpha]
                    del converted[std_alpha]
            if found_alpha is not None:
                converted[f"{prefix}.alpha"] = found_alpha
        return converted

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        weights_sd = self._state_dict_from_standard_packed_lora_keys(weights_sd)
        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1:]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            te1_loras = [
                lora for lora in self.text_encoder_loras
                if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)
            ]
            if len(te1_loras) > 0:
                logger.info(f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}")
                params, descriptions = assemble_params(te1_loras, text_encoder_lr[0], loraplus_ratio)
                all_params.extend(params)
                lr_descriptions.extend(["textencoder 1" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    @staticmethod
    def _pad_tensor_dim(tensor: torch.Tensor, dim: int, padded_size: int) -> torch.Tensor:
        dim = dim % tensor.ndim
        if tensor.size(dim) == padded_size:
            return tensor
        if tensor.size(dim) > padded_size:
            raise ValueError(
                f"cannot pad dim {dim} from {tensor.size(dim)} down to {padded_size}"
            )
        out_shape = list(tensor.shape)
        out_shape[dim] = padded_size
        out = tensor.new_zeros(out_shape)
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(0, tensor.size(dim))
        out[tuple(index)] = tensor
        return out

    @staticmethod
    def _trim_tensor_dim(tensor: torch.Tensor, dim: int, original_size: int) -> torch.Tensor:
        dim = dim % tensor.ndim
        if tensor.size(dim) < original_size:
            raise ValueError(
                f"cannot trim dim {dim} of size {tensor.size(dim)} to original_size={original_size}"
            )
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(0, original_size)
        return tensor[tuple(index)].contiguous()

    def gather_tp_lora_weights(self, trim_padding: bool = False) -> None:
        """Gather sharded LoRA weights from all TP ranks so rank 0 holds the full LoRA.

        Column-parallel LoRA: lora_up is sharded on dim 0 (out_features/tp) → gather dim 0
        Row-parallel LoRA:    lora_down is sharded on dim 1 (in_features/tp) → gather dim 1
                              (lora_down.weight shape is (rank, in_features/tp))

        When trim_padding=True, gathered padded shards are trimmed to the
        original unpadded base-module shape for standard LoRA checkpoint saving.
        scatter_tp_lora_weights() accepts either trimmed or padded full weights.
        """
        import torch.distributed as dist

        for lora in self.text_encoder_loras + self.unet_loras:
            if isinstance(lora, PackedColumnParallelLoRAModule) and lora._tp_group is not None:
                org_module = lora.org_module_ref[0]
                original_part = int(getattr(org_module, "original_part_size", 0) or 0)
                for up in lora.lora_up:
                    w = up.weight.data
                    orig_device = w.device
                    w_c = w.contiguous().cuda()
                    gathered = [torch.zeros_like(w_c) for _ in range(lora._tp_group.size())]
                    dist.all_gather(gathered, w_c, group=lora._tp_group)
                    full = torch.cat(gathered, dim=0).to(orig_device)
                    if trim_padding and original_part:
                        full = self._trim_tensor_dim(full, 0, original_part)
                    up.weight.data = full

            elif isinstance(lora, ColumnParallelLoRAModule) and lora._tp_group is not None:
                # lora_up.weight: (out_features/tp, lora_dim) → gather dim 0
                w = lora.lora_up.weight.data
                orig_device = w.device
                # cuda_direct requires CUDA tensors; weights may be on CPU before
                # the network moves to GPU (e.g. during verify checks).
                w_c = w.contiguous().cuda()
                gathered = [torch.zeros_like(w_c) for _ in range(lora._tp_group.size())]
                dist.all_gather(gathered, w_c, group=lora._tp_group)
                full = torch.cat(gathered, dim=0).to(orig_device)
                org_module = lora.org_module_ref[0]
                if trim_padding and hasattr(org_module, "original_out_features"):
                    full = self._trim_tensor_dim(full, 0, int(org_module.original_out_features))
                lora.lora_up.weight.data = full

            elif isinstance(lora, RowParallelLoRAModule) and lora._tp_group is not None:
                # lora_down.weight: (lora_dim, in_features/tp) → gather dim 1
                w = lora.lora_down.weight.data
                orig_device = w.device
                w_c = w.contiguous().cuda()
                gathered = [torch.zeros_like(w_c) for _ in range(lora._tp_group.size())]
                dist.all_gather(gathered, w_c, group=lora._tp_group)
                full = torch.cat(gathered, dim=1).to(orig_device)
                org_module = lora.org_module_ref[0]
                if trim_padding and hasattr(org_module, "original_in_features"):
                    full = self._trim_tensor_dim(full, 1, int(org_module.original_in_features))
                lora.lora_down.weight.data = full

    def scatter_tp_lora_weights(self) -> None:
        """Re-shard LoRA weights back to per-rank slices after gather_tp_lora_weights().

        Column-parallel LoRA: lora_up.weight (D_out, lora_dim) → slice dim 0 → (D_out/tp, lora_dim)
        Row-parallel LoRA:    lora_down.weight (lora_dim, D_in) → slice dim 1 → (lora_dim, D_in/tp)
        """
        import torch.distributed as dist

        for lora in self.text_encoder_loras + self.unet_loras:
            if isinstance(lora, PackedColumnParallelLoRAModule) and lora._tp_group is not None:
                tp = lora._tp_group.size()
                rank = dist.get_rank(group=lora._tp_group)
                org_module = lora.org_module_ref[0]
                padded_part = int(getattr(org_module, "padded_part_size", 0) or 0)
                for up in lora.lora_up:
                    w = up.weight.data
                    if padded_part and w.shape[0] < padded_part:
                        w = self._pad_tensor_dim(w, 0, padded_part)
                    chunk = w.shape[0] // tp
                    up.weight.data = w[rank * chunk:(rank + 1) * chunk].contiguous()

            elif isinstance(lora, ColumnParallelLoRAModule) and lora._tp_group is not None:
                tp = lora._tp_group.size()
                rank = dist.get_rank(group=lora._tp_group)
                w = lora.lora_up.weight.data          # (D_out, lora_dim) after gather
                org_module = lora.org_module_ref[0]
                padded_out = int(getattr(org_module, "padded_out_features", w.shape[0]))
                if w.shape[0] < padded_out:
                    w = self._pad_tensor_dim(w, 0, padded_out)
                chunk = padded_out // tp
                lora.lora_up.weight.data = w[rank * chunk:(rank + 1) * chunk].contiguous()

            elif isinstance(lora, RowParallelLoRAModule) and lora._tp_group is not None:
                tp = lora._tp_group.size()
                rank = dist.get_rank(group=lora._tp_group)
                w = lora.lora_down.weight.data        # (lora_dim, D_in) after gather
                org_module = lora.org_module_ref[0]
                padded_in = int(getattr(org_module, "padded_in_features", w.shape[1]))
                if w.shape[1] < padded_in:
                    w = self._pad_tensor_dim(w, 1, padded_in)
                chunk = padded_in // tp
                lora.lora_down.weight.data = w[:, rank * chunk:(rank + 1) * chunk].contiguous()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self._state_dict_to_standard_packed_lora_keys(self.state_dict())

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
