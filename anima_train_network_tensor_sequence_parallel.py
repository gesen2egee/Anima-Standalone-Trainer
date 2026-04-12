# Anima LoRA training script — TP (Tensor Parallel) + SP (Sequence Parallel)
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
#      (QKV is NOT fused — keeps standard key names for LoRA compatibility)
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
from typing import Union

import torch
import torch.distributed as dist

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import anima_train_utils, train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Import the base trainer — it brings in train_network.NetworkTrainer too
import anima_train_network
from anima_train_network import AnimaNetworkTrainer


def _make_anima_lora_tp_spec(sequence_parallel: bool = False, use_llm_adapter: bool = False):
    """TP spec for Anima LoRA — targets UNFUSED projections.

    Unlike the full-finetune TP spec (which fuses q/k/v into qkv_proj),
    LoRA must keep separate q_proj/k_proj/v_proj so that the saved LoRA
    has standard key names compatible with non-TP inference.

    This costs 3 all-gathers per self-attn block instead of 1, but for
    LoRA the extra communication is negligible relative to the matmuls.

    LLM Adapter (use_llm_adapter=True): shards the 6-block cross-attention
    transformer that bridges Qwen3 → T5 space.  Adapter self-attn uses SP
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
        entries.update({
            # LLM Adapter self-attn: target sequence (B, T, D) — SP applies
            "llm_adapter.blocks.*.self_attn.q_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.k_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.v_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.o_proj":  row(sp),
            # LLM Adapter cross-attn: q from target (SP-sharded), kv from Qwen3 context (replicated)
            "llm_adapter.blocks.*.cross_attn.q_proj": col(sp),
            "llm_adapter.blocks.*.cross_attn.k_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.v_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.o_proj": row(sp),
            # LLM Adapter MLP: Sequential[0]=Linear(D→4D), [1]=GELU, [2]=Linear(4D→D)
            "llm_adapter.blocks.*.mlp.0":             col(sp),
            "llm_adapter.blocks.*.mlp.2":             row(sp),
        })
    return wdp.ParallelSpec(entries)


def _fixup_attention_heads_for_tp(model: torch.nn.Module, tp_size: int) -> int:
    """After apply_parallelism(), divide n_heads by tp_size in every Attention module.

    ColumnParallelLinear shards the OUTPUT dimension (features), so q/k/v proj
    output D/tp features per rank.  The Attention.compute_qkv rearrange uses
    h=self.n_heads hardcoded; with D/tp features and full n_heads it fails.
    Fix: set n_heads = n_heads // tp_size so each rank treats its shard as
    (n_heads/tp, head_dim) — mathematically equivalent to head-parallel attention.

    Returns the number of Attention modules updated.
    """
    from library.anima_models import Attention, LLMAdapterAttention
    from wd_parallel.layers import ColumnParallelLinear

    updated = 0
    for module in model.modules():
        # Covers both main-model Attention (uses einops rearrange) and
        # LLMAdapterAttention (uses .view) — both break the same way after TP sharding.
        if isinstance(module, (Attention, LLMAdapterAttention)) and isinstance(module.q_proj, ColumnParallelLinear):
            assert module.n_heads % tp_size == 0, (
                f"{type(module).__name__}: n_heads={module.n_heads} not divisible by tp_size={tp_size}"
            )
            module.n_heads = module.n_heads // tp_size
            updated += 1
    return updated


def _tag_tp_lora_params(network: torch.nn.Module) -> int:
    """Tag only the truly TP-sharded LoRA parameters with _tp_sharded=True.

    - ColumnParallelLoRAModule.lora_up   is sharded across TP ranks (tagged)
    - ColumnParallelLoRAModule.lora_down is replicated         (NOT tagged)
    - RowParallelLoRAModule.lora_down    is sharded across TP ranks (tagged)
    - RowParallelLoRAModule.lora_up      is replicated         (NOT tagged)

    sync_replicated_grads() skips tagged params and all-reduces the rest,
    so replicated LoRA params get proper gradient synchronization.

    Returns the number of tagged parameters.
    """
    from networks.lora_anima import ColumnParallelLoRAModule, RowParallelLoRAModule
    tagged = 0
    for module in network.modules():
        if isinstance(module, ColumnParallelLoRAModule):
            for p in module.lora_up.parameters():
                p._tp_sharded = True
                tagged += 1
        elif isinstance(module, RowParallelLoRAModule):
            for p in module.lora_down.parameters():
                p._tp_sharded = True
                tagged += 1
    return tagged


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

    # ----- override: model loading + TP sharding -----

    def load_target_model(self, args, weight_dtype, accelerator):
        """Load model normally, then apply TP sharding (no QKV fusion).

        QKV is NOT fused for LoRA — this keeps standard key names so the
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

            # Run model-forward check with mock data after sharding
            from tp_sp_verify import run_all_checks as _tp_verify
            _tp_verify(dit=dit, network=None, groups=self.tp_groups, use_sp=self.use_sp)

        return model_type, text_encoders, vae, dit

    # ----- override: skip DDP wrapping for TP models -----

    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        """For TP: move to device without Accelerator DDP wrapping.

        Three things that would break with TP + Accelerator DDP:
          1. accelerator.prepare(unet) → wraps frozen base model with DDP (wrong)
          2. accelerator.prepare(network) → wraps LoRA with DDP, broadcasts
             rank 0's params → destroys TP-sharded LoRA weights
          3. accelerator.prepare(dataloader) → distributed sampler gives each
             rank different batches → TP needs SAME batch on all ranks

        Fix: set distributed_type=NO so Accelerator skips DDP wrapping
        AND distributed sampling.  TP handles distribution itself; mixed
        precision and gradient accumulation still work with NO.

        This override runs BEFORE the network/dataloader prepare calls
        (line 953 in NetworkTrainer.train), so the change takes effect.
        """
        if not self.tp_active:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # Prevent DDP wrapping of both base model AND LoRA network.
        # Also prevents distributed DataLoader sampler — TP needs all ranks
        # to see the SAME batch (unlike DDP which splits batches).
        from accelerate.utils import DistributedType
        real_rank = dist.get_rank() if dist.is_initialized() else 0
        accelerator.state.distributed_type = DistributedType.NO
        # process_index is already set to 0 on all ranks by the __main__
        accelerator.state.local_process_index = real_rank
        accelerator.state.num_processes = 1  # each rank acts as single-GPU for Accelerator
        logger.info(f"TP active — disabled Accelerator DDP (rank={real_rank}, TP handles distribution)")

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
                # Sharded params: each rank holds a slice → need all-reduce max.
                # Replicated params: identical on all ranks → all-reduce max is a no-op but harmless.
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
                #     → squared norms across ranks sum to the full-weight squared norm
                #     → must all-reduce (SUM) so every rank uses the global total.
                #
                #   Replicated (_tp_sharded absent/False):  every rank holds an
                #     identical copy (sync_replicated_grads already ran).
                #     → simply adding all ranks' contributions would count the
                #       same gradient tp_size times, inflating the norm by
                #       sqrt(tp_size) and over-clipping by that factor.
                #     → include only locally; no cross-rank reduction needed.
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

        # Handle unsloth offload checkpointing
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        # Block swap support
        if self.is_swapping_blocks:
            unet.move_to_device_except_swap_blocks(accelerator.device)
            unet.prepare_block_swap_before_forward()
        else:
            unet.to(accelerator.device)

        return unet

    # ----- override: TP-aware LoRA gradient sync -----

    def all_reduce_network(self, accelerator, network):
        """Sync LoRA gradients across TP ranks.

        TP-sharded LoRA params (on Column/RowParallelLinear) are tagged
        _tp_sharded — each rank trains its own shard, no sync needed.
        Replicated LoRA params (on LayerNorm, AdaLN, embeddings) need
        gradient averaging across TP ranks, especially with SP where
        each rank sees different sequence tokens.
        """
        if not self.tp_active:
            super().all_reduce_network(accelerator, network)
            return

        import wd_parallel as wdp
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

        if not self.tp_active:
            return

        tagged = _tag_tp_lora_params(network)
        logger.info(f"Tagged {tagged} TP-sharded LoRA parameters (will not be gradient-synced)")

        # Check LoRA tagging is correct
        from tp_sp_verify import run_all_checks as _tp_verify
        _tp_verify(dit=None, network=network, groups=self.tp_groups, use_sp=self.use_sp)

        # Wrap save_weights to gather sharded LoRA weights from all TP ranks
        # before saving, then scatter them back so training can continue.
        #
        # Because we set process_index=0 on ALL ranks (see prepare_unet_with_accelerator),
        # the inherited training loop's save_model() is entered by every rank.
        # gather/scatter use all_gather (a collective) — all ranks MUST call them.
        # Only rank 0 actually writes the file.
        tp_rank = self.tp_groups.tp_rank
        _orig_save = network.save_weights

        def _tp_save_weights(file, dtype, metadata):
            # Gather shards → full weights (for save), then re-shard so training continues.
            # gather_tp_lora_weights() mutates weight.data in-place; scatter restores shards.
            try:
                network.gather_tp_lora_weights()
                # Only rank 0 writes the file (all ranks have identical gathered weights)
                if tp_rank == 0:
                    _orig_save(file, dtype, metadata)
            finally:
                # Barrier so no rank proceeds until save is done
                if dist.is_initialized():
                    dist.barrier()
                # Restore sharded weights — training must continue with per-rank slices
                network.scatter_tp_lora_weights()

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

        if tp_will_be_active and getattr(args, 'save_state', False):
            raise ValueError(
                "save_state is not supported with TP LoRA training. "
                "With TP, all ranks act as main process for save coordination; "
                "Accelerator save_state would produce conflicting state files."
            )

        if tp_will_be_active and getattr(args, 'huggingface_repo_id', None):
            raise ValueError(
                "huggingface_repo_id is not supported with TP LoRA training. "
                "All ranks would attempt to upload simultaneously."
            )


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    # Start with ALL regular LoRA training args (inherits train_network + anima)
    parser = anima_train_network.setup_parser()

    # TP/SP-specific args
    parser.add_argument(
        "--tp_degree", type=int, default=2,
        help="Tensor Parallel degree. Must match --nproc_per_node in torchrun. (default: 2)",
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", default=False,
        help="Enable Sequence Parallel (SP) alongside TP. Halves activation memory "
             "at the cost of two extra collectives per block.",
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

    tp_degree = getattr(args, "tp_degree", 1)
    use_sp = getattr(args, "sequence_parallel", False) and tp_degree > 1

    # ---------------------------------------------------------------------------
    # Initialize TP BEFORE trainer.train() — must happen before Accelerator
    # calls dist.init_process_group.  wdp.init_dist() sets up the process group;
    # Accelerator will detect it's already initialized and skip re-init.
    # ---------------------------------------------------------------------------
    tp_groups = None
    tp_config = None
    if tp_degree > 1:
        # Add wd_parallel to path
        _WDP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wd_parallel")
        if _WDP_PATH not in sys.path:
            sys.path.insert(0, _WDP_PATH)

        import wd_parallel as wdp

        tp_backend = wdp.activate_backend()    # detect cuda_direct or gloo; sets USE_LIBUV on Windows
        dist.init_process_group(backend=tp_backend)
        tp_config = wdp.ParallelConfig(tp=True, sp=use_sp)
        tp_groups = wdp.init_dist(tp_config)
        logger.info(f"TP initialized: rank={tp_groups.tp_rank}/{tp_groups.tp_size}, sp={use_sp}, backend={tp_backend}")

        # Run collective + layer math checks immediately after TP init
        # (before model load — fast, catches backend/collective bugs early)
        from tp_sp_verify import run_all_checks as _tp_verify
        _tp_verify(dit=None, network=None, groups=tp_groups, use_sp=use_sp)

        # Monkeypatch prepare_accelerator so the Accelerator is created with
        # process_index=0 on ALL ranks.  train_network.py caches
        # `is_main_process = accelerator.is_main_process` immediately after
        # creating the Accelerator — by the time our prepare_unet override
        # runs, that cached boolean is already frozen.  This patch ensures
        # the cache captures True on every rank, so all ranks enter save_model()
        # and participate in the collective gather/scatter inside _tp_save_weights.
        _orig_prepare_accelerator = train_util.prepare_accelerator
        def _tp_prepare_accelerator(args):
            acc = _orig_prepare_accelerator(args)
            acc.state.process_index = 0
            return acc
        train_util.prepare_accelerator = _tp_prepare_accelerator

    # Create trainer and inject TP state
    trainer = AnimaNetworkTrainerTPSP()
    trainer.tp_config = tp_config
    trainer.tp_groups = tp_groups
    trainer.use_sp = use_sp

    # Run the full training loop — inherited from NetworkTrainer
    trainer.train(args)

    # Cleanup TP
    if tp_groups is not None:
        import wd_parallel as wdp
        wdp.destroy_dist()
