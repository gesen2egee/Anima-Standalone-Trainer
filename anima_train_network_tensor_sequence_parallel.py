# Anima LoRA training script - TP (Tensor Parallel) + SP (Sequence Parallel)
#
# Inherits from AnimaNetworkTrainer so that ALL regular LoRA training features
# (validation, differential output preservation, masked loss, all config flags,
# resume, sample images, etc.) work identically.  Only the methods that need
# TP/SP awareness are overridden.
#
# Launch with torchrun:
#   torchrun --nproc_per_node=2 anima_train_network_tensor_sequence_parallel.py --tp_degree 2 [other args]
#
# The TP sharding happens inside load_target_model():
#   1. Parent loads the model normally (cpu)
#   2. wd_parallel shards projection weights across TP ranks
#      (QKV is NOT fused - keeps standard key names for LoRA compatibility)
#   3. LoRA network creation (in the parent's train()) then wraps the sharded layers
#      using TP-aware LoRA modules that handle SP communication automatically
#
# Key design decisions:
#   - Accelerator distributed_type is set to NO so DDP doesn't wrap the LoRA
#     network or shard the dataloader (TP needs same batch on all ranks)
#   - LoRA gradient sync uses wdp.sync_replicated_grads() instead of DDP
#   - Saved LoRA is gathered from all TP ranks into standard format (rank 0 saves)
#   - Sample generation is skipped (TP forward needs all ranks in collectives)

import argparse
import os
import sys
import time
from typing import Union

import torch
import torch.distributed as dist

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import anima_train_utils, train_util, save_utils, huggingface_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ROOT = r"C:\Anima\split_files"
DEFAULT_DIT_PATH = os.path.join(DEFAULT_MODEL_ROOT, "diffusion_models", "anima-preview.safetensors")
DEFAULT_QWEN3_PATH = os.path.join(DEFAULT_MODEL_ROOT, "text_encoders", "qwen_3_06b_base.safetensors")
DEFAULT_VAE_PATH = os.path.join(DEFAULT_MODEL_ROOT, "vae", "qwen_image_vae.safetensors")


def _ensure_wd_parallel_importable():
    """Make the sibling wd_parallel repo importable for this standalone trainer."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    wdp_path = os.path.abspath(os.path.join(repo_root, "..", "wd_parallel"))
    if wdp_path not in sys.path:
        sys.path.insert(0, wdp_path)


def _apply_default_model_paths(args):
    """Fill Anima model paths from C:\\Anima\\split_files when omitted."""
    defaults = {
        "dit_path": DEFAULT_DIT_PATH,
        "qwen3_path": DEFAULT_QWEN3_PATH,
        "vae_path": DEFAULT_VAE_PATH,
    }
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
        if not os.path.exists(getattr(args, name)):
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")
    return args


# Import the base trainer - it brings in train_network.NetworkTrainer too
import anima_train_network
from anima_train_network import AnimaNetworkTrainer


def _make_anima_lora_tp_spec(sequence_parallel: bool = False, use_llm_adapter: bool = False):
    """TP spec for Anima LoRA - targets UNFUSED projections.

    Unlike the full-finetune TP spec (which fuses q/k/v into qkv_proj),
    LoRA must keep separate q_proj/k_proj/v_proj so that the saved LoRA
    has standard key names compatible with non-TP inference.

    This costs 3 all-gathers per self-attn block instead of 1, but for
    LoRA the extra communication is negligible relative to the matmuls.

    LLM Adapter (use_llm_adapter=True): shards the 6-block cross-attention
    transformer that bridges Qwen3 -> T5 space.  Adapter self-attn uses SP
    (target sequence is sharded); adapter cross-attn KV uses
    sequence_parallel=False (Qwen3 context is replicated across TP ranks).
    """
    import wd_parallel as wdp
    sp = sequence_parallel
    col = lambda sp_flag: wdp.ColumnParallelSpec(sequence_parallel=sp_flag, seq_dim=1)
    row = lambda sp_flag: wdp.RowParallelSpec(sequence_parallel=sp_flag, seq_dim=1)
    entries = {
        # Self-attention: individual projections (no QKV fusion)
        "blocks.*.self_attn.q_proj":       col(sp),
        "blocks.*.self_attn.k_proj":       col(sp),
        "blocks.*.self_attn.v_proj":       col(sp),
        "blocks.*.self_attn.output_proj":  row(sp),
        # Cross-attention: k/v use replicated context (never SP)
        "blocks.*.cross_attn.q_proj":      col(sp),
        "blocks.*.cross_attn.k_proj":      col(False),
        "blocks.*.cross_attn.v_proj":      col(False),
        "blocks.*.cross_attn.output_proj": row(sp),
        # MLP
        "blocks.*.mlp.layer1":             col(sp),
        "blocks.*.mlp.layer2":             row(sp),
    }
    if use_llm_adapter:
        # The LLM Adapter's T5 target sequence is REPLICATED on all TP ranks -
        # it is never scattered by SP.  Using col(sp) / row(sp) here would make
        # ColumnParallelLinear all-gather on seq_dim before the matmul, doubling
        # the sequence (each rank provides its full copy -> 2x tokens) and causing
        # shape mismatches downstream.  Always use col(False) / row(False) so the
        # adapter runs plain column-parallel (no sequence gather/scatter).
        entries.update({
            "llm_adapter.blocks.*.self_attn.q_proj":  col(False),
            "llm_adapter.blocks.*.self_attn.k_proj":  col(False),
            "llm_adapter.blocks.*.self_attn.v_proj":  col(False),
            "llm_adapter.blocks.*.self_attn.o_proj":  row(False),
            "llm_adapter.blocks.*.cross_attn.q_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.k_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.v_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.o_proj": row(False),
            # MLP: Sequential[0]=Linear(D->4D), [2]=Linear(4D->D)
            "llm_adapter.blocks.*.mlp.0":             col(False),
            "llm_adapter.blocks.*.mlp.2":             row(False),
        })
    return wdp.ParallelSpec(entries)


def _fixup_attention_heads_for_tp(model: torch.nn.Module, tp_size: int) -> int:
    """After apply_parallelism(), divide n_heads by tp_size in every Attention module.

    ColumnParallelLinear shards the OUTPUT dimension (features), so q/k/v proj
    output D/tp features per rank.  The Attention.compute_qkv rearrange uses
    h=self.n_heads hardcoded; with D/tp features and full n_heads it fails.
    Fix: set n_heads = n_heads // tp_size so each rank treats its shard as
    (n_heads/tp, head_dim) - mathematically equivalent to head-parallel attention.

    Returns the number of Attention modules updated.
    """
    from library.anima_models import Attention, LLMAdapterAttention
    from wd_parallel.layers import ColumnParallelLinear

    updated = 0
    for module in model.modules():
        # Covers both main-model Attention (uses einops rearrange) and
        # LLMAdapterAttention (uses .view) - both break the same way after TP sharding.
        if isinstance(module, (Attention, LLMAdapterAttention)) and isinstance(module.q_proj, ColumnParallelLinear):
            assert module.n_heads % tp_size == 0, (
                f"{type(module).__name__}: n_heads={module.n_heads} not divisible by tp_size={tp_size}"
            )
            module.n_heads = module.n_heads // tp_size
            updated += 1
    return updated


def _tag_tp_lora_params(network: torch.nn.Module) -> tuple[int, int]:
    """Tag TP-sharded and SP-partial LoRA params for wd_parallel grad sync.

    Column-parallel LoRA: lora_up is sharded; lora_down is replicated but
    receives partial output-shard gradients, so it needs SUM semantics.

    Row-parallel LoRA: lora_down is sharded; lora_up is replicated but
    receives partial feature/sequence gradients, so it needs SUM semantics.
    """
    from networks.lora_anima import ColumnParallelLoRAModule, RowParallelLoRAModule

    sharded = 0
    partial = 0
    for lora in network.unet_loras:
        if isinstance(lora, ColumnParallelLoRAModule):
            for p in lora.lora_up.parameters():
                p._tp_sharded = True
                sharded += 1
            for p in lora.lora_down.parameters():
                p._tp_sharded = False
                p._tp_partial_grad = True
                partial += 1
        elif isinstance(lora, RowParallelLoRAModule):
            for p in lora.lora_down.parameters():
                p._tp_sharded = True
                sharded += 1
            for p in lora.lora_up.parameters():
                p._tp_sharded = False
                p._tp_partial_grad = True
                partial += 1
    return sharded, partial


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

class AnimaNetworkTrainerTPSP(AnimaNetworkTrainer):
    """LoRA trainer with Tensor Parallel + optional Sequence Parallel.

    Inherits the full training loop, validation, sampling, all config flags,
    and all LoRA features from AnimaNetworkTrainer / NetworkTrainer.
    Only overrides what TP/SP requires.
    """

    def __init__(self):
        super().__init__()
        self.tp_config = None
        self.tp_groups = None
        self.tp_active = False
        self.use_sp = False
        self._tp_step = 0
        self._nan_grad_reported = False
        self._tp_diag_path = None
        self._tp_diag_header_written = False
        self._tp_last_args = None

    def _tp_rank(self):
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _tp_diag(self, args, message, *, all_ranks=False, to_logger=False):
        if not self.tp_active and self.tp_groups is None:
            return
        rank = self._tp_rank()
        if rank != 0 and not all_ranks:
            return
        if self._tp_diag_path is None:
            base = getattr(args, "logging_dir", None) or getattr(args, "output_dir", None) or os.getcwd()
            os.makedirs(base, exist_ok=True)
            self._tp_diag_path = os.path.join(base, f"tp_sp_diagnostics_rank{rank}.log")
        prefix = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{prefix}] rank={rank} {message}"
        if not self._tp_diag_header_written:
            with open(self._tp_diag_path, "a", encoding="utf-8") as f:
                f.write("--- TP/SP diagnostic session ---\n")
            self._tp_diag_header_written = True
        with open(self._tp_diag_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if to_logger and rank == 0:
            logger.info(line)

    # ----- override: model loading + TP sharding -----

    def load_target_model(self, args, weight_dtype, accelerator):
        """Load model normally, then apply TP sharding (no QKV fusion).

        QKV is NOT fused for LoRA - this keeps standard key names so the
        saved LoRA is compatible with non-TP inference pipelines.
        """
        model_type, text_encoders, vae, dit = super().load_target_model(args, weight_dtype, accelerator)

        if self.tp_groups is not None and self.tp_groups.tp_size > 1:
            import wd_parallel as wdp

            # Apply TP sharding with unfused spec (individual q/k/v projections)
            tp_spec = _make_anima_lora_tp_spec(self.use_sp, use_llm_adapter=getattr(dit, "use_llm_adapter", False))
            dit = wdp.apply_parallelism(dit, tp_spec, self.tp_config, self.tp_groups)
            self.tp_active = True

            # Attention modules still use the original n_heads after sharding.
            # Divide by tp_size so each rank computes on its n_heads//tp heads.
            n_attn_fixed = _fixup_attention_heads_for_tp(dit, self.tp_groups.tp_size)

            # SP scatter/gather: set the process group so MiniTrainDIT.forward
            # scatters x along H before the block loop and gathers back after.
            if self.use_sp:
                dit._tp_sp_group = self.tp_groups.tp

            logger.info(
                f"TP sharding applied: tp_degree={self.tp_groups.tp_size}, sp={self.use_sp}, "
                f"llm_adapter={getattr(dit, 'use_llm_adapter', False)}, "
                f"attention_modules_patched={n_attn_fixed}"
            )
            self._tp_diag(
                args,
                f"startup backend={dist.get_backend()} tp_degree={self.tp_groups.tp_size} sp={self.use_sp} "
                f"llm_adapter={getattr(dit, 'use_llm_adapter', False)} attention_modules_patched={n_attn_fixed} "
                f"device={accelerator.device} dtype={weight_dtype}",
                all_ranks=True,
                to_logger=True,
            )

            # Optional full-model forward check. It is useful for debugging, but it
            # can be too memory-heavy before block swap/checkpointing is active.
            if getattr(args, "tp_verify_model_forward", False):
                from tp_sp_verify import run_all_checks as _tp_verify
                _tp_verify(dit=dit, network=None, groups=self.tp_groups, use_sp=self.use_sp)
            else:
                self._tp_diag(args, "skip full-model verify before training", all_ranks=True, to_logger=True)

        return model_type, text_encoders, vae, dit

    # ----- override: skip DDP wrapping for TP models -----

    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        """For TP: move to device without Accelerator DDP wrapping.

        Three things that would break with TP + Accelerator DDP:
          1. accelerator.prepare(unet) -> wraps frozen base model with DDP (wrong)
          2. accelerator.prepare(network) -> wraps LoRA with DDP, broadcasts
             rank 0's params -> destroys TP-sharded LoRA weights
          3. accelerator.prepare(dataloader) -> distributed sampler gives each
             rank different batches -> TP needs SAME batch on all ranks

        Fix: set distributed_type=NO so Accelerator skips DDP wrapping
        AND distributed sampling.  TP handles distribution itself; mixed
        precision and gradient accumulation still work with NO.

        This override runs BEFORE the network/dataloader prepare calls
        (line 953 in NetworkTrainer.train), so the change takes effect.
        """
        if not self.tp_active:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        self._tp_diag(args, "begin prepare_unet_with_accelerator", all_ranks=True, to_logger=True)

        # Prevent DDP wrapping of both base model AND LoRA network.
        # Also prevents distributed DataLoader sampler - TP needs all ranks
        # to see the SAME batch (unlike DDP which splits batches).
        from accelerate.utils import DistributedType
        real_rank = dist.get_rank() if dist.is_initialized() else 0
        accelerator.state.distributed_type = DistributedType.NO
        # process_index is already set to 0 on all ranks by the __main__
        accelerator.state.local_process_index = real_rank
        accelerator.state.num_processes = 1  # each rank acts as single-GPU for Accelerator
        logger.info(f"TP active - disabled Accelerator DDP (rank={real_rank}, TP handles distribution)")

        # Monkeypatch clip_grad_norm_ to compute global norm across TP ranks.
        tp_group = self.tp_groups.tp

        def _tp_clip_grad_norm_(
            parameters: Union[torch.Tensor, list],
            max_norm: float,
            norm_type: float = 2.0,
        ):
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            if len(parameters) == 0:
                return torch.tensor(0.0)

            device = parameters[0].grad.device
            norm_type = float(norm_type)

            if norm_type == float('inf'):
                # inf norm: global max across all params and all TP ranks.
                # Sharded params: each rank holds a slice -> need all-reduce max.
                # Replicated params: identical on all ranks -> all-reduce max is a no-op but harmless.
                local_max = torch.tensor(
                    max(p.grad.detach().abs().max().item() for p in parameters),
                    device=device,
                )
                dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
                total_norm = local_max
            else:
                # Lp norm: sum of p-th powers, then take p-th root.
                #
                # Two kinds of params must be treated differently to avoid
                # counting the same gradient contribution multiple times:
                #
                #   TP-sharded (_tp_sharded=True):  each rank holds a unique shard
                #     -> squared norms across ranks sum to the full-weight squared norm
                #     -> must all-reduce (SUM) so every rank uses the global total.
                #
                #   Replicated (_tp_sharded absent/False):  every rank holds an
                #     identical copy (sync_replicated_grads already ran).
                #     -> simply adding all ranks' contributions would count the
                #       same gradient tp_size times, inflating the norm by
                #       sqrt(tp_size) and over-clipping by that factor.
                #     -> include only locally; no cross-rank reduction needed.
                sharded_acc = torch.zeros(1, device=device)
                replicated_acc = torch.zeros(1, device=device)
                for p in parameters:
                    contrib = p.grad.detach().norm(norm_type).pow(norm_type)
                    if getattr(p, '_tp_sharded', False):
                        sharded_acc += contrib
                    else:
                        replicated_acc += contrib

                # All-reduce only the sharded contribution
                dist.all_reduce(sharded_acc, op=dist.ReduceOp.SUM, group=tp_group)
                total_norm = (sharded_acc + replicated_acc).pow(1.0 / norm_type)

            # Clip using the global norm (identical on all ranks after all-reduce)
            clip_coef = max_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for p in parameters:
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

            return total_norm.item()

        accelerator.clip_grad_norm_ = _tp_clip_grad_norm_

        if not getattr(accelerator.prepare, "_tp_stage_wrapped", False):
            _orig_accelerator_prepare = accelerator.prepare

            def _tp_accelerator_prepare(*objects, **kwargs):
                names = ",".join(type(o).__name__ for o in objects)
                self._tp_diag(args, f"begin accelerator.prepare objects={names}", all_ranks=True, to_logger=True)
                out = _orig_accelerator_prepare(*objects, **kwargs)
                self._tp_diag(args, f"end accelerator.prepare objects={names}", all_ranks=True, to_logger=True)
                return out

            _tp_accelerator_prepare._tp_stage_wrapped = True
            accelerator.prepare = _tp_accelerator_prepare

        # Handle unsloth offload checkpointing
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        # Block swap support
        if self.is_swapping_blocks:
            unet.move_to_device_except_swap_blocks(accelerator.device)
            unet.prepare_block_swap_before_forward()
        else:
            unet.to(accelerator.device)

        self._tp_diag(args, "end prepare_unet_with_accelerator", all_ranks=True, to_logger=True)
        return unet

    # ----- override: broadcast batch from rank 0 so all TP ranks see same data -----

    @staticmethod
    def _broadcast_tensor(t, tp_group, device):
        """Broadcast a tensor from rank 0 to all TP ranks, handling shape mismatches."""
        if t is None:
            return None
        t = t.to(device)
        shape_t = torch.tensor(list(t.shape), dtype=torch.int64, device=device)
        dist.broadcast(shape_t, src=0, group=tp_group)
        canonical = torch.Size(shape_t.tolist())
        if t.shape != canonical:
            t = torch.zeros(canonical, dtype=t.dtype, device=device)
        t = t.contiguous()
        dist.broadcast(t, src=0, group=tp_group)
        return t

    def process_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler,
                      vae_dtype, weight_dtype, accelerator, args,
                      text_encoding_strategy, tokenize_strategy,
                      is_train=True, train_text_encoder=True, train_unet=True):
        """Broadcast all per-sample batch tensors from rank 0 to all TP ranks.

        TP requires all ranks to process the SAME batch. Without synchronization,
        each rank independently samples from the DataLoader (same seed but diverged
        Python RNG state) and loads different images from different resolution
        buckets. This causes ColumnParallelLinear allgathers to receive
        unequal-sized inputs (out-of-bounds SHM read / cudaErrorInvalidValue).
        """
        if self.tp_active and dist.is_initialized():
            self._tp_last_args = args
            tp_group = self.tp_groups.tp
            dev = accelerator.device

            # Latents (main tensor - shape differs across resolution buckets)
            if "latents" in batch and batch["latents"] is not None:
                batch["latents"] = self._broadcast_tensor(batch["latents"], tp_group, dev)

            # Per-sample loss weights
            if "loss_weights" in batch and batch["loss_weights"] is not None:
                batch["loss_weights"] = self._broadcast_tensor(batch["loss_weights"], tp_group, dev)

            # Cached text encoder outputs
            te_list = batch.get("text_encoder_outputs_list", None)
            if te_list is not None:
                batch["text_encoder_outputs_list"] = [
                    self._broadcast_tensor(te, tp_group, dev) for te in te_list
                ]

            # Alpha masks (optional)
            if "alpha_masks" in batch and batch["alpha_masks"] is not None:
                batch["alpha_masks"] = self._broadcast_tensor(batch["alpha_masks"], tp_group, dev)

        self._tp_step += 1
        loss = super().process_batch(
            batch, text_encoders, unet, network, vae, noise_scheduler,
            vae_dtype, weight_dtype, accelerator, args,
            text_encoding_strategy, tokenize_strategy,
            is_train=is_train, train_text_encoder=train_text_encoder,
            train_unet=train_unet,
        )

        # Loss diagnostics: every step goes to a sidecar file; first NaN/Inf also hits tqdm.
        if self.tp_active:
            loss_val = loss.detach().float().item()
            finite_loss = loss_val == loss_val and loss_val not in (float('inf'), float('-inf'))
            lat = batch.get("latents", None)
            lat_shape = tuple(lat.shape) if lat is not None else None
            mem = ""
            if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
                mem = f" cuda_mem_alloc_mb={torch.cuda.memory_allocated() / (1024 ** 2):.1f}"
            self._tp_diag(args, f"step={self._tp_step} loss={loss_val:.8g} finite={finite_loss} latent_shape={lat_shape}{mem}", all_ranks=True)
            if self._tp_rank() == 0 and not finite_loss:
                from tqdm import tqdm
                tqdm.write(
                    f"[TP NaN] step {self._tp_step}: forward loss={loss_val}  "
                    f"latent_shape={lat_shape}"
                )

        return loss

    # ----- override: per-step weight check -----

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        """Check LoRA weights for NaN/Inf before each training step.

        Runs after optimizer.zero_grad() of the *previous* step (i.e. after the
        previous optimizer.step()), so catches weights that were corrupted by
        the optimizer update.  All ranks check their own weight slices because
        TP-sharded params differ per rank.
        """
        if not self.tp_active or not is_train:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        step = self._tp_step + 1  # _tp_step increments inside process_batch; this is the upcoming step

        bad_weights = []
        for name, p in network.named_parameters():
            if not torch.isfinite(p.data).all():
                bad_weights.append(
                    f"{name}(shape={tuple(p.shape)}, "
                    f"nan={torch.isnan(p.data).sum().item()}, "
                    f"inf={torch.isinf(p.data).sum().item()})"
                )
        if bad_weights:
            from tqdm import tqdm
            tqdm.write(
                f"[TP CHECK step {step} rank {rank}] NaN/Inf in LoRA WEIGHTS:\n"
                + "\n".join(f"  {s}" for s in bad_weights[:10])
                + ("\n  ... (truncated)" if len(bad_weights) > 10 else "")
            )

    # ----- override: TP-aware LoRA gradient sync -----

    def all_reduce_network(self, accelerator, network):
        """Sync LoRA gradients across TP ranks.

        TP-sharded LoRA params (on Column/RowParallelLinear) are tagged
        _tp_sharded - each rank trains its own shard, no sync needed.
        Replicated LoRA params (on LayerNorm, AdaLN, embeddings) need
        gradient averaging across TP ranks, especially with SP where
        each rank sees different sequence tokens.
        """
        if not self.tp_active:
            super().all_reduce_network(accelerator, network)
            return

        # Gradient diagnostics on rank 0: log L2 norm every step + report first
        # NaN/Inf occurrence.  Norm tracking reveals gradients growing toward NaN
        # before the loss itself goes NaN.
        if dist.get_rank() == 0:
            total_sq = 0.0
            max_abs  = 0.0
            nan_params = []
            for name, p in network.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if not torch.isfinite(g).all():
                    nan_params.append(
                        f"{name}(shape={tuple(g.shape)}, "
                        f"nan={torch.isnan(g).sum().item()}, "
                        f"inf={torch.isinf(g).sum().item()})"
                    )
                else:
                    total_sq += g.norm(2).item() ** 2
                    cur_max = g.abs().max().item()
                    if cur_max > max_abs:
                        max_abs = cur_max

            from tqdm import tqdm
            grad_norm = total_sq**0.5
            tqdm.write(
                f"[TP GRAD step {self._tp_step}] "
                f"norm={grad_norm:.4f}  max={max_abs:.4f}"
                + (f"  NaN/Inf in {len(nan_params)} param(s)" if nan_params else "")
            )
            self._tp_diag(
                self._tp_last_args or argparse.Namespace(),
                f"step={self._tp_step} grad_before_sync_norm={grad_norm:.8g} grad_before_sync_max={max_abs:.8g} nonfinite_grad_params={len(nan_params)}",
            )

            if nan_params and not self._nan_grad_reported:
                tqdm.write(
                    f"[TP NaN] step {self._tp_step}: NaN/Inf gradients BEFORE sync:\n"
                    + "\n".join(f"  {s}" for s in nan_params[:10])
                    + ("\n  ... (truncated)" if len(nan_params) > 10 else "")
                )
                self._nan_grad_reported = True

        import wd_parallel as wdp
        # On the first step, reset the NaN diagnostic counter in collectives so
        # any events triggered by the verify check (before training) don't suppress
        # the first real training NaN report.
        if self._tp_step == 1:
            wdp.reset_nan_diagnostics()
        # sync_replicated_grads skips _tp_sharded params automatically
        wdp.sync_replicated_grads(network, self.tp_groups.tp)

    # ----- override: post-process network to tag TP LoRA params + hook save -----

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """Called after LoRA network is created but before apply_to().

        1. Tag LoRA params on TP layers so sync_replicated_grads skips them
        2. Wrap save_weights so TP shards are gathered before saving
           (produces standard LoRA format compatible with non-TP inference)
        """
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        self._tp_lora_network = network

        if not self.tp_active:
            return

        sharded, partial = _tag_tp_lora_params(network)
        logger.info(f"Tagged {sharded} TP-sharded LoRA parameters and {partial} SP-partial replicated LoRA parameters")
        self._tp_diag(args, f"lora_params sharded={sharded} sp_partial_replicated={partial}", all_ranks=True, to_logger=True)

        # Stage markers for quick TP/SP smoke runs. Exit 137 gives no Python
        # traceback, so these breadcrumbs show the last successful setup step.
        def _wrap_stage(method_name):
            original = getattr(network, method_name, None)
            if original is None or getattr(original, "_tp_stage_wrapped", False):
                return

            def wrapped(*a, **kw):
                self._tp_diag(args, f"begin {method_name}", all_ranks=True, to_logger=True)
                out = original(*a, **kw)
                self._tp_diag(args, f"end {method_name}", all_ranks=True, to_logger=True)
                return out

            wrapped._tp_stage_wrapped = True
            setattr(network, method_name, wrapped)

        for _method in [
            "apply_to",
            "enable_gradient_checkpointing",
            "prepare_optimizer_params",
            "prepare_optimizer_params_with_multiple_te_lrs",
            "prepare_grad_etc",
        ]:
            _wrap_stage(_method)

        # Check LoRA tagging is correct
        from tp_sp_verify import run_all_checks as _tp_verify
        _tp_verify(dit=None, network=network, groups=self.tp_groups, use_sp=self.use_sp)

        # Reset diagnostic hook flags - tp_sp_verify calls LoRA module forward()
        # in training mode, which sets _hooks_registered=True on the class before
        # actual training starts, silencing all hooks for the real training run.
        from networks.lora_anima import ColumnParallelLoRAModule, RowParallelLoRAModule
        ColumnParallelLoRAModule._hooks_registered = False
        RowParallelLoRAModule._hooks_registered = False

        # Wrap save_weights to gather sharded LoRA weights from all TP ranks
        # before saving, then scatter them back so training can continue.
        #
        # Because we set process_index=0 on ALL ranks (see prepare_unet_with_accelerator),
        # the inherited training loop's save_model() is entered by every rank.
        # gather/scatter use all_gather (a collective) - all ranks MUST call them.
        # Only rank 0 actually writes the file.
        tp_rank = self.tp_groups.tp_rank
        tp_size = self.tp_groups.tp_size
        _orig_save = network.save_weights

        def _scatter_tp_lora_weights_local() -> None:
            # Avoid torch.distributed metadata calls here; CUDA Direct can hang in
            # group operations after the checkpoint write. We already know rank/size.
            for lora in network.text_encoder_loras + network.unet_loras:
                if isinstance(lora, ColumnParallelLoRAModule) and lora._tp_group is not None:
                    w = lora.lora_up.weight.data
                    chunk = w.shape[0] // tp_size
                    lora.lora_up.weight.data = w[tp_rank * chunk:(tp_rank + 1) * chunk].contiguous()

                elif isinstance(lora, RowParallelLoRAModule) and lora._tp_group is not None:
                    w = lora.lora_down.weight.data
                    chunk = w.shape[1] // tp_size
                    lora.lora_down.weight.data = w[:, tp_rank * chunk:(tp_rank + 1) * chunk].contiguous()

        def _tp_save_weights(file, dtype, metadata):
            # Gather shards -> full weights (for save), then re-shard so training continues.
            # gather_tp_lora_weights() mutates weight.data in-place; scatter restores shards.
            try:
                network.gather_tp_lora_weights()
                # Only rank 0 writes the file (all ranks have identical gathered weights)
                if tp_rank == 0:
                    _orig_save(file, dtype, metadata)
            finally:
                # Restore sharded weights - training must continue with per-rank slices
                _scatter_tp_lora_weights_local()

        network.save_weights = _tp_save_weights

    # ----- override: sample images (TP requires all ranks in forward) -----

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        """Sample generation with TP-sharded models.

        TP forward passes require ALL ranks to participate in collectives.
        ALL ranks run EVERY prompt together; only rank 0 saves the image
        (via save_image=False on non-zero ranks).
        """
        if not self.tp_active:
            return super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

        if args.sample_prompts is None:
            return

        # Check timing (same logic as anima_train_utils.sample_images)
        if global_step == 0:
            if not args.sample_at_first:
                return
        else:
            if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
                return
            if args.sample_every_n_epochs is not None:
                if epoch is None or epoch % args.sample_every_n_epochs != 0:
                    return
            elif args.sample_every_n_steps is not None:
                if global_step % args.sample_every_n_steps != 0 or epoch is not None:
                    return

        tp_rank = self.tp_groups.tp_rank
        logger.info(f"[TP rank {tp_rank}] Generating sample images at step {global_step}")

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        dit = accelerator.unwrap_model(unet)
        if qwen3_te is not None:
            qwen3_te = accelerator.unwrap_model(qwen3_te)
        sample_dtype = next(dit.parameters()).dtype

        prompts = train_util.load_prompts(args.sample_prompts)
        save_dir = os.path.join(args.output_dir, "sample")
        if tp_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # Save RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        org_vae_device = next(vae.parameters()).device
        vae.to(accelerator.device)
        vae_scale_gpu = [t.to(accelerator.device) for t in self.vae_scale]

        from library.anima_train_utils import _sample_image_inference

        lora_network = getattr(self, "_tp_lora_network", None)
        original_lora_dtypes = []
        if lora_network is not None:
            for param in lora_network.parameters():
                if param.is_floating_point():
                    original_lora_dtypes.append((param, param.dtype))
                    param.data = param.data.to(dtype=sample_dtype)

        try:
            with torch.no_grad(), accelerator.autocast():
                for prompt_dict in prompts:
                    # ALL ranks run forward (TP collectives need all ranks)
                    # Only rank 0 saves the image file
                    _sample_image_inference(
                        accelerator, args, dit, qwen3_te, vae, vae_scale_gpu,
                        self.tokenize_strategy, self.text_encoding_strategy,
                        save_dir, prompt_dict, epoch, global_step,
                        self.sample_prompts_te_outputs, None,
                        save_image=(tp_rank == 0),
                    )
        finally:
            for param, dtype in original_lora_dtypes:
                param.data = param.data.to(dtype=dtype)

            vae.to(org_vae_device)
            clean_memory_on_device(accelerator.device)

            # Restore RNG state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

        if dist.is_initialized():
            dist.barrier()

    # ----- assert: validate TP args -----

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        # NOTE: assert_extra_args is called before load_target_model, so self.tp_active
        # is not yet set. Use tp_groups to detect TP mode at validation time.
        tp_will_be_active = self.tp_groups is not None and self.tp_groups.tp_size > 1

        if tp_will_be_active and getattr(args, 'blockwise_fused_optimizers', False):
            raise ValueError("blockwise_fused_optimizers is not supported with TP+SP LoRA training")

        if tp_will_be_active and getattr(args, 'scale_weight_norms', None):
            raise ValueError(
                "scale_weight_norms is not supported with TP LoRA training. "
                "apply_max_norm_regularization computes norms on sharded weight "
                "slices, which gives incorrect results when the full weight is "
                "split across TP ranks."
            )

        if (
            tp_will_be_active
            and getattr(args, 'huggingface_repo_id', None)
            and not getattr(args, 'save_state_to_huggingface', False)
        ):
            raise ValueError(
                "huggingface_repo_id is not supported with TP LoRA training. "
                "All ranks would attempt to upload simultaneously. Set "
                "--save_state_to_huggingface only when uploading TP state folders."
            )


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    # Start with ALL regular LoRA training args (inherits train_network + anima)
    parser = anima_train_network.setup_parser()

    # TP/SP-specific args. This script intentionally runs TP+SP together.
    parser.add_argument(
        "--tp_degree", type=int, default=2,
        help="Tensor Parallel degree. Must match --nproc_per_node in torchrun. (default: 2)",
    )
    parser.add_argument(
        "--tp_backend", type=str, default="auto", choices=["auto", "cuda_direct", "nccl"],
        help="Distributed backend for TP+SP. Use cuda_direct on native Windows, nccl on WSL/Linux.",
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", default=True,
        help="Kept for config compatibility; TP mode always enables SP in this script.",
    )
    parser.add_argument(
        "--no_sequence_parallel", action="store_true",
        help="Rejected intentionally: this trainer is TP+SP-only.",
    )
    parser.add_argument(
        "--tp_verify_model_forward", action="store_true",
        help="Run the expensive full-DiT TP/SP forward diagnostic before training.",
    )
    return parser


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    args = _apply_default_model_paths(args)

    tp_degree = int(getattr(args, "tp_degree", 1))
    if tp_degree <= 1:
        raise ValueError("anima_train_network_tensor_sequence_parallel.py is TP+SP-only; use tp_degree >= 2")
    if getattr(args, "no_sequence_parallel", False):
        raise ValueError("--no_sequence_parallel is not supported here; this trainer intentionally runs TP+SP together")
    use_sp = True
    args.sequence_parallel = True

    _ensure_wd_parallel_importable()
    import wd_parallel as wdp

    tp_backend = wdp.activate_backend(getattr(args, "tp_backend", "auto"))
    dist.init_process_group(backend=tp_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if world_size != tp_degree:
        raise ValueError(f"tp_degree={tp_degree} must match torchrun world_size={world_size}")

    tp_config = wdp.ParallelConfig(tp=True, sp=True, tp_degree=tp_degree)
    tp_groups = wdp.init_dist(tp_config)
    logger.info(f"TP+SP initialized: rank={tp_groups.tp_rank}/{tp_groups.tp_size}, backend={tp_backend}")
    logger.info(f"Using model paths: dit={args.dit_path}, qwen3={args.qwen3_path}, vae={args.vae_path}")
    save_utils.install_parallel_state_wrappers(
        train_util_module=train_util,
        huggingface_util_module=huggingface_util,
        parallel_rank=tp_groups.tp_rank,
        parallel_size=tp_groups.tp_size,
        process_group=tp_groups.tp,
        backend=tp_backend,
        logger=logger,
        mode="tp_sp",
    )

    # Fast sanity checks before the expensive model load.
    from tp_sp_verify import run_all_checks as _tp_verify
    _tp_verify(dit=None, network=None, groups=tp_groups, use_sp=True)

    # Make every rank participate in save hooks that contain TP collectives.
    # num_processes=1 must match here: caching uses i%num_processes!=process_index to shard work,
    # and with process_index=0 on all ranks, leaving num_processes>1 drops ~50% of images uncached.
    _orig_prepare_accelerator = train_util.prepare_accelerator
    def _tp_prepare_accelerator(args):
        acc = _orig_prepare_accelerator(args)
        # The base trainer guards checkpoint/state/sample hooks with
        # accelerator.is_main_process. In TP/SP those hooks include collectives,
        # so every TP rank must enter them. Spoof both global and local process
        # identity; Accelerate exposes several cached/proxied views of these.
        acc.state.process_index = 0
        acc.state.local_process_index = 0
        acc.state.num_processes = 1
        try:
            acc.process_index = 0
            acc.local_process_index = 0
            acc.num_processes = 1
        except Exception:
            pass
        logger.info(
            "TP accelerator spoof: "
            f"rank={tp_groups.tp_rank}, "
            f"process_index={acc.process_index}, "
            f"local_process_index={acc.local_process_index}, "
            f"num_processes={acc.num_processes}, "
            f"is_main={acc.is_main_process}, "
            f"is_local_main={acc.is_local_main_process}"
        )
        return acc
    train_util.prepare_accelerator = _tp_prepare_accelerator

    # Create trainer and inject TP state
    trainer = AnimaNetworkTrainerTPSP()
    trainer.tp_config = tp_config
    trainer.tp_groups = tp_groups
    trainer.use_sp = use_sp

    # Run the full training loop - inherited from NetworkTrainer
    trainer.train(args)

    # Cleanup TP
    if tp_groups is not None:
        import wd_parallel as wdp
        wdp.destroy_dist()
