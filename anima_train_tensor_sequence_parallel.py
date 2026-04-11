# Anima full finetune training script — TP (Tensor Parallel) + SP (Sequence Parallel)
#
# Thin subclass of AnimaTrainer (anima_train.py).  Only the ~6 hook methods that
# need TP/SP awareness are overridden; the full training loop is inherited.
#
# Launch with torchrun:
#   torchrun --nproc_per_node=2 anima_train_tensor_sequence_parallel.py --tp_degree 2 [other args]

import os
import sys
import torch
import torch.distributed as dist

from library.device_utils import init_ipex
init_ipex()

from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# wd_parallel — resolve path relative to this script
_WDP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wd_parallel")
if _WDP_PATH not in sys.path:
    sys.path.insert(0, _WDP_PATH)
try:
    import wd_parallel as wdp
    _WDP_AVAILABLE = True
except ImportError:
    logger.warning("wd_parallel not found — TP disabled. Clone C:/wd_parallel next to this repo.")
    _WDP_AVAILABLE = False

# Import everything from the base trainer
import library.train_util as train_util
from anima_train import AnimaTrainer, setup_parser as _base_setup_parser


# ---------------------------------------------------------------------------
# Helper functions — QKV fusion, unfusion, TP spec
# ---------------------------------------------------------------------------

def fuse_qkv_for_tp(model: torch.nn.Module) -> None:
    """Fuse separate Q/K/V projections into combined linear layers, in-place.

    Self-attention:  q_proj + k_proj + v_proj → qkv_proj  (3*inner_dim output)
    Cross-attention: k_proj + v_proj           → kv_proj   (2*inner_dim output)
                     q_proj stays separate     (different input: x vs context)

    TP benefit: cuts self-attn from 3 ColumnParallel all-gathers to 1,
    cross-attn from 3 to 2.  With 28 blocks: 84 → 42 all-gathers for QKV.
    """
    import types
    from library.anima_models import Attention

    def _fused_self_attn_compute_qkv(self, x, context=None, rope_emb=None):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        n_h = q.shape[-1] // self.head_dim
        from einops import rearrange
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_h, d=self.head_dim), (q, k, v))
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        if rope_emb is not None:
            from library.anima_models import apply_rotary_pos_emb
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=False)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=False)
        return q, k, v

    def _fused_cross_attn_compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        ctx = x if context is None else context
        k, v = self.kv_proj(ctx).chunk(2, dim=-1)
        n_h = q.shape[-1] // self.head_dim
        from einops import rearrange
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=n_h, d=self.head_dim), (q, k, v))
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        return q, k, v

    fused_count = 0
    for block in model.blocks:
        sa, ca = block.self_attn, block.cross_attn

        # Self-attn: q/k/v → qkv
        fused_w = torch.cat([sa.q_proj.weight.data, sa.k_proj.weight.data, sa.v_proj.weight.data], dim=0)
        sa.qkv_proj = torch.nn.Linear(sa.q_proj.in_features, fused_w.shape[0], bias=False,
                                       device=sa.q_proj.weight.device, dtype=sa.q_proj.weight.dtype)
        sa.qkv_proj.weight = torch.nn.Parameter(fused_w)
        del sa.q_proj, sa.k_proj, sa.v_proj
        sa.compute_qkv = types.MethodType(_fused_self_attn_compute_qkv, sa)

        # Cross-attn: k/v → kv
        fused_kv_w = torch.cat([ca.k_proj.weight.data, ca.v_proj.weight.data], dim=0)
        ca.kv_proj = torch.nn.Linear(ca.k_proj.in_features, fused_kv_w.shape[0], bias=False,
                                      device=ca.k_proj.weight.device, dtype=ca.k_proj.weight.dtype)
        ca.kv_proj.weight = torch.nn.Parameter(fused_kv_w)
        del ca.k_proj, ca.v_proj
        ca.compute_qkv = types.MethodType(_fused_cross_attn_compute_qkv, ca)

        fused_count += 1

    logger.info(f"QKV fusion: {fused_count} blocks — self-attn 3→1, cross-attn 3→2 all-gathers for QKV.")


def unfuse_qkv_from_tp(model: torch.nn.Module) -> None:
    """Reverse fuse_qkv_for_tp — splits fused weights back to separate projections.

    Call before every model save so checkpoints are compatible with anima_train.py.
    Call fuse_qkv_for_tp() again after saving to resume fast training.
    """
    import types
    from library.anima_models import Attention
    original_compute_qkv = Attention.compute_qkv

    for block in model.blocks:
        sa, ca = block.self_attn, block.cross_attn

        # Un-fuse self-attn: qkv_proj → q/k/v
        qkv_w = sa.qkv_proj.weight.data
        inner, in_f = qkv_w.shape[0] // 3, qkv_w.shape[1]
        dev, dt = qkv_w.device, qkv_w.dtype
        sa.q_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.k_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.v_proj = torch.nn.Linear(in_f, inner, bias=False, device=dev, dtype=dt)
        sa.q_proj.weight = torch.nn.Parameter(qkv_w[:inner].clone())
        sa.k_proj.weight = torch.nn.Parameter(qkv_w[inner:2*inner].clone())
        sa.v_proj.weight = torch.nn.Parameter(qkv_w[2*inner:].clone())
        del sa.qkv_proj
        sa.compute_qkv = types.MethodType(original_compute_qkv, sa)

        # Un-fuse cross-attn: kv_proj → k/v
        kv_w = ca.kv_proj.weight.data
        inner_kv, in_f_kv = kv_w.shape[0] // 2, kv_w.shape[1]
        ca.k_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
        ca.v_proj = torch.nn.Linear(in_f_kv, inner_kv, bias=False, device=dev, dtype=dt)
        ca.k_proj.weight = torch.nn.Parameter(kv_w[:inner_kv].clone())
        ca.v_proj.weight = torch.nn.Parameter(kv_w[inner_kv:].clone())
        del ca.kv_proj
        ca.compute_qkv = types.MethodType(original_compute_qkv, ca)


def _make_anima_tp_spec(sequence_parallel: bool = False, use_llm_adapter: bool = False) -> "wdp.ParallelSpec":
    """ParallelSpec for MiniTrainDIT after fuse_qkv_for_tp() has been applied."""
    sp  = sequence_parallel
    col = lambda f: wdp.ColumnParallelSpec(sequence_parallel=f, seq_dim=1)
    row = lambda f: wdp.RowParallelSpec(sequence_parallel=f,    seq_dim=1)
    entries = {
        "blocks.*.self_attn.qkv_proj":     col(sp),
        "blocks.*.self_attn.output_proj":  row(sp),
        "blocks.*.cross_attn.q_proj":      col(sp),
        "blocks.*.cross_attn.kv_proj":     col(False),  # context is replicated
        "blocks.*.cross_attn.output_proj": row(sp),
        "blocks.*.mlp.layer1":             col(sp),
        "blocks.*.mlp.layer2":             row(sp),
    }
    if use_llm_adapter:
        entries.update({
            "llm_adapter.blocks.*.self_attn.q_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.k_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.v_proj":  col(sp),
            "llm_adapter.blocks.*.self_attn.o_proj":  row(sp),
            "llm_adapter.blocks.*.cross_attn.q_proj": col(sp),
            "llm_adapter.blocks.*.cross_attn.k_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.v_proj": col(False),
            "llm_adapter.blocks.*.cross_attn.o_proj": row(sp),
            "llm_adapter.blocks.*.mlp.0":             col(sp),
            "llm_adapter.blocks.*.mlp.2":             row(sp),
        })
    return wdp.ParallelSpec(entries)


# ---------------------------------------------------------------------------
# TP-aware trainer subclass
# ---------------------------------------------------------------------------

class AnimaTrainerTPSP(AnimaTrainer):

    def __init__(self):
        self.tp_groups  = None
        self.tp_config  = None
        self.tp_active  = False
        self.use_sp     = False
        self.train_dit  = True  # updated in apply_model_parallelism

    # --- Hook 1: TP initialisation (before dataset/model setup) ---

    def on_train_begin(self, args):
        tp_degree = getattr(args, 'tp_degree', 1)
        self.use_sp = getattr(args, 'sequence_parallel', False) and tp_degree > 1

        if tp_degree > 1 and _WDP_AVAILABLE:
            tp_backend = wdp.activate_backend()
            dist.init_process_group(backend=tp_backend)
            self.tp_config = wdp.ParallelConfig(tp=True, sp=self.use_sp)
            self.tp_groups = wdp.init_dist(self.tp_config)
            self.tp_active = True
            logger.info(
                f"TP initialized: rank={self.tp_groups.tp_rank}/{self.tp_groups.tp_size}, "
                f"sp={self.use_sp}, backend={tp_backend}"
            )
        elif tp_degree > 1:
            raise RuntimeError("wd_parallel is required for TP but could not be imported.")

    # --- Hook 2: QKV fusion + TP sharding (after DiT load, before optimizer) ---

    def apply_model_parallelism(self, args, dit):
        if not self.tp_active:
            return dit

        self.train_dit = args.learning_rate != 0
        use_llm_adapter = getattr(dit, 'use_llm_adapter', False)

        fuse_qkv_for_tp(dit)
        tp_spec = _make_anima_tp_spec(self.use_sp, use_llm_adapter=use_llm_adapter)
        dit = wdp.apply_parallelism(dit, tp_spec, self.tp_config, self.tp_groups)

        n_params = sum(p.numel() for p in dit.parameters())
        n_sharded = sum(p.numel() for p in dit.parameters() if getattr(p, '_tp_sharded', False))
        logger.info(
            f"TP sharding applied: tp_degree={self.tp_groups.tp_size}, sp={self.use_sp}, "
            f"params={n_params:,}, sharded={n_sharded:,}"
        )
        return dit

    # --- Hook 3: skip Accelerator DDP wrapping for TP ---

    def prepare_dit_with_accelerator(self, accelerator, dit, is_swapping_blocks):
        if not self.tp_active:
            return super().prepare_dit_with_accelerator(accelerator, dit, is_swapping_blocks)

        # TP handles its own communication — DDP wrapping would conflict.
        if not is_swapping_blocks:
            dit = dit.to(accelerator.device)
        else:
            dit.move_to_device_except_swap_blocks(accelerator.device)
        return dit

    # --- Hook 4: sync non-sharded param gradients across TP ranks ---

    def sync_gradients(self, dit):
        if self.tp_active and self.tp_groups.tp_size > 1:
            wdp.sync_replicated_grads(dit, self.tp_groups.tp)

    # --- Hook 5: unfuse QKV before save ---

    def before_save(self, dit):
        if self.tp_active and self.train_dit:
            unfuse_qkv_from_tp(dit)

    # --- Hook 6: re-fuse QKV after save so training continues fast ---

    def after_save(self, dit, train_dit):
        if self.tp_active and train_dit:
            fuse_qkv_for_tp(dit)
            dit.requires_grad_(train_dit)

    # --- Hook 7: final unfuse before end-of-training saves ---

    def on_train_end(self, dit):
        if self.tp_active and self.train_dit:
            unfuse_qkv_from_tp(dit)

    # --- Hook 8: destroy TP process group ---

    def on_cleanup(self):
        if self.tp_active:
            wdp.destroy_dist()


# ---------------------------------------------------------------------------
# Parser — base args + TP/SP additions
# ---------------------------------------------------------------------------

def setup_parser() -> "argparse.ArgumentParser":
    parser = _base_setup_parser()
    parser.add_argument(
        "--tp_degree", type=int, default=1,
        help="Tensor Parallel degree. 1=disabled. Requires torchrun --nproc_per_node=N.",
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", default=False,
        help="Enable Sequence Parallel alongside TP (requires --tp_degree >= 2).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    AnimaTrainerTPSP().train(args)
