"""
Lumina LoRA Training Debug Tracker
===================================
Tracks VRAM usage, tensor statistics, and block device placement throughout the
training loop to help diagnose VRAM spikes, unexpected tensor movements, and
other anomalies specific to Lumina training.

Usage — add to lumina_train_network.py:

    from library.lumina_debug_tracker import LuminaDebugTracker, DebugTrackerConfig

    # In LuminaNetworkTrainer.__init__:
    self.debug_tracker = LuminaDebugTracker(DebugTrackerConfig(
        enabled=True,
        log_every_n_steps=10,
        log_vram=True,
        log_block_devices=True,
        log_tensor_stats=True,
        log_attention_sizes=True,
        log_on_phase_change=True,  # always log at sampling/checkpoint boundaries
    ))

    # In on_step_start:
    self.debug_tracker.on_step_start(global_step, unet, phase="train")

    # In get_noise_pred_and_target (after computing model_pred):
    self.debug_tracker.on_forward_end(global_step, noisy_model_input, model_pred, timesteps)

    # In sample_images (before and after):
    self.debug_tracker.on_phase_boundary(global_step, "sampling_start")
    # ... sampling ...
    self.debug_tracker.on_phase_boundary(global_step, "sampling_end")

    # In save checkpoint area (wrap via on_phase_boundary):
    self.debug_tracker.on_phase_boundary(global_step, "checkpoint_save")
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebugTrackerConfig:
    enabled: bool = True

    # How often to log during normal training steps (0 = never log during steps)
    log_every_n_steps: int = 10

    # Always log at phase boundaries (sampling start/end, checkpoint saves)
    log_on_phase_change: bool = True

    # What to log
    log_vram: bool = True           # CUDA memory: allocated, reserved, peak
    log_block_devices: bool = True  # Which DiT blocks are on GPU vs CPU
    log_tensor_stats: bool = True   # noisy_model_input and model_pred shape/dtype/device/stats
    log_attention_sizes: bool = True  # Estimated attention intermediate size based on input shape
    log_lora_norms: bool = False    # LoRA weight norms (expensive, off by default)

    # VRAM spike detection: log extra detail if VRAM jumps by more than this (MB)
    vram_spike_threshold_mb: float = 200.0

    # Log to file in addition to logger
    log_file: Optional[str] = None  # e.g. "lumina_debug.log"


# ─────────────────────────────────────────────────────────────────────────────
# VRAM snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VRAMSnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def capture() -> "VRAMSnapshot":
        if not torch.cuda.is_available():
            return VRAMSnapshot(0, 0, 0, 0)
        mem = torch.cuda.memory_stats()
        to_mb = 1 / (1024 ** 2)
        return VRAMSnapshot(
            allocated_mb=torch.cuda.memory_allocated() * to_mb,
            reserved_mb=torch.cuda.memory_reserved() * to_mb,
            peak_allocated_mb=mem.get("allocated_bytes.all.peak", 0) * to_mb,
            peak_reserved_mb=mem.get("reserved_bytes.all.peak", 0) * to_mb,
        )

    def delta(self, prev: "VRAMSnapshot") -> Tuple[float, float]:
        """Return (delta_allocated_mb, delta_reserved_mb) relative to prev."""
        return (
            self.allocated_mb - prev.allocated_mb,
            self.reserved_mb - prev.reserved_mb,
        )

    def format(self, prev: Optional["VRAMSnapshot"] = None) -> str:
        parts = [
            f"alloc={self.allocated_mb:.0f}MB",
            f"reserved={self.reserved_mb:.0f}MB",
            f"peak_alloc={self.peak_allocated_mb:.0f}MB",
            f"peak_reserved={self.peak_reserved_mb:.0f}MB",
        ]
        if prev is not None:
            da, dr = self.delta(prev)
            sign_a = "+" if da >= 0 else ""
            sign_r = "+" if dr >= 0 else ""
            parts.append(f"Δalloc={sign_a}{da:.0f}MB Δreserved={sign_r}{dr:.0f}MB")
        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Block device snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _block_device_summary(model: nn.Module) -> Dict[str, str]:
    """
    Return a dict mapping block name → device string for each block in the DiT.

    Handles NextDiT structure: self.layers, self.context_refiner, self.noise_refiner.
    Falls back to inspecting first parameter of each block.
    """
    result: Dict[str, str] = {}

    def _first_param_device(module: nn.Module) -> str:
        try:
            p = next(module.parameters())
            return str(p.device)
        except StopIteration:
            return "?"

    for attr in ("layers", "context_refiner", "noise_refiner"):
        blocks = getattr(model, attr, None)
        if blocks is None:
            continue
        for i, block in enumerate(blocks):
            result[f"{attr}[{i}]"] = _first_param_device(block)

    # Also track top-level embedders / output layers
    for attr in ("x_embedder", "t_embedder", "cap_embedder", "final_layer", "norm_final"):
        mod = getattr(model, attr, None)
        if mod is not None:
            result[attr] = _first_param_device(mod)

    return result


def _format_block_devices(snap: Dict[str, str], prev: Optional[Dict[str, str]] = None) -> str:
    """Format block device snapshot, highlighting changes from prev."""
    lines: List[str] = []

    # Summarise layers compactly: show ranges that are on the same device
    layer_devices = {k: v for k, v in snap.items() if k.startswith("layers[")}
    if layer_devices:
        # Build compact run-length encoding
        items = sorted(layer_devices.items(), key=lambda x: int(x[0].split("[")[1].rstrip("]")))
        runs: List[Tuple[int, int, str]] = []  # (start_idx, end_idx, device)
        for name, dev in items:
            idx = int(name.split("[")[1].rstrip("]"))
            if runs and runs[-1][2] == dev:
                runs[-1] = (runs[-1][0], idx, dev)
            else:
                runs.append((idx, idx, dev))
        parts = [f"[{s}-{e}]={d}" if s != e else f"[{s}]={d}" for s, e, d in runs]
        lines.append("  layers: " + " ".join(parts))

    # Other blocks line by line
    other = {k: v for k, v in snap.items() if not k.startswith("layers[")}
    for name, dev in other.items():
        changed = prev is not None and prev.get(name) != dev
        marker = " *** CHANGED" if changed else ""
        lines.append(f"  {name}: {dev}{marker}")

    # Check if any layers changed
    if prev is not None:
        changed_layers = [k for k in layer_devices if prev.get(k) != layer_devices[k]]
        if changed_layers:
            lines.append(f"  *** {len(changed_layers)} layer(s) changed device: {changed_layers[:8]}")

    return "\n".join(lines) if lines else "  (no blocks found)"


# ─────────────────────────────────────────────────────────────────────────────
# Tensor stats
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_stats(t: torch.Tensor, name: str) -> str:
    if t is None:
        return f"  {name}: None"
    with torch.no_grad():
        try:
            f = t.float()
            return (
                f"  {name}: shape={list(t.shape)} dtype={t.dtype} device={t.device} "
                f"min={f.min().item():.4f} max={f.max().item():.4f} "
                f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
                f"has_nan={torch.isnan(f).any().item()} has_inf={torch.isinf(f).any().item()}"
            )
        except Exception as e:
            return f"  {name}: shape={list(t.shape)} dtype={t.dtype} device={t.device} [stats error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Attention size estimate
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_attention_memory(noisy_input: torch.Tensor, cap_len: int = 256) -> str:
    """
    Estimate attention intermediate memory for NextDiT joint attention.
    noisy_input shape: (B, C, H, W) where image tokens = (H/2)*(W/2) after patchify with patch_size=2.
    """
    if noisy_input is None or noisy_input.ndim != 4:
        return "  attention estimate: n/a"
    B, C, H, W = noisy_input.shape
    # NextDiT patchifies with patch_size=2: each 2x2 spatial region → 1 token
    img_tokens = (H // 2) * (W // 2)
    # Joint attention: image tokens + caption tokens
    seq_len = img_tokens + cap_len
    n_heads = 24  # NextDiT default
    bytes_per_elem = 2  # bf16/fp16
    # QK^T matrix per head: seq_len x seq_len
    attn_matrix_mb = B * n_heads * seq_len * seq_len * bytes_per_elem / (1024 ** 2)
    # Q, K, V projections stored: 3 * B * seq_len * head_dim (head_dim=128 for 3072-dim, 24-head)
    head_dim = 128
    qkv_mb = B * 3 * n_heads * seq_len * head_dim * bytes_per_elem / (1024 ** 2)
    total_mb = attn_matrix_mb + qkv_mb
    return (
        f"  attention estimate: img_tokens={img_tokens} cap_tokens={cap_len} "
        f"seq_len={seq_len}  attn_matrix={attn_matrix_mb:.0f}MB  qkv={qkv_mb:.0f}MB  "
        f"total_per_layer={total_mb:.0f}MB  (x26 layers ≈ {total_mb*26:.0f}MB)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker class
# ─────────────────────────────────────────────────────────────────────────────

class LuminaDebugTracker:
    def __init__(self, config: DebugTrackerConfig = DebugTrackerConfig()):
        self.cfg = config
        self._prev_vram: Optional[VRAMSnapshot] = None
        self._prev_block_snap: Optional[Dict[str, str]] = None
        self._file_handler: Optional[object] = None
        self._step_count = 0

        if config.enabled and config.log_file:
            try:
                import logging as _logging
                fh = _logging.FileHandler(config.log_file, mode="a", encoding="utf-8")
                fh.setFormatter(_logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
                self._file_handler = fh
            except Exception as e:
                logger.warning(f"[LuminaDebug] Could not open log file {config.log_file}: {e}")

    def _emit(self, msg: str):
        logger.info(msg)
        if self._file_handler:
            try:
                import logging as _logging
                record = _logging.LogRecord(
                    name="lumina_debug", level=_logging.INFO,
                    pathname="", lineno=0, msg=msg, args=(), exc_info=None
                )
                self._file_handler.emit(record)
            except Exception:
                pass

    def _should_log(self, step: int) -> bool:
        if not self.cfg.enabled:
            return False
        n = self.cfg.log_every_n_steps
        return n > 0 and step % n == 0

    def _snapshot(self, step: int, label: str, model: Optional[nn.Module] = None,
                  noisy_input: Optional[torch.Tensor] = None,
                  model_pred: Optional[torch.Tensor] = None,
                  timesteps: Optional[torch.Tensor] = None,
                  cap_len: int = 256):
        lines = [f"\n{'='*60}", f"[LuminaDebug] step={step}  phase={label}"]

        if self.cfg.log_vram:
            vram = VRAMSnapshot.capture()
            lines.append(f"  VRAM: {vram.format(self._prev_vram)}")
            # Spike detection
            if self._prev_vram is not None:
                _, dr = vram.delta(self._prev_vram)
                if dr > self.cfg.vram_spike_threshold_mb:
                    lines.append(
                        f"  *** VRAM SPIKE DETECTED: reserved grew by {dr:.0f}MB "
                        f"(threshold={self.cfg.vram_spike_threshold_mb:.0f}MB)"
                    )
            self._prev_vram = vram

        if self.cfg.log_block_devices and model is not None:
            try:
                snap = _block_device_summary(model)
                lines.append("  Block devices:")
                lines.append(_format_block_devices(snap, self._prev_block_snap))
                self._prev_block_snap = snap
            except Exception as e:
                lines.append(f"  Block devices: [error: {e}]")

        if self.cfg.log_tensor_stats:
            if noisy_input is not None:
                lines.append(_tensor_stats(noisy_input, "noisy_model_input"))
            if model_pred is not None:
                lines.append(_tensor_stats(model_pred, "model_pred"))
            if timesteps is not None:
                lines.append(f"  timesteps: {timesteps.tolist()[:8]} dtype={timesteps.dtype}")

        if self.cfg.log_attention_sizes and noisy_input is not None:
            lines.append(_estimate_attention_memory(noisy_input, cap_len))

        self._emit("\n".join(lines))

    # ── Public API ────────────────────────────────────────────────────────────

    def on_step_start(self, step: int, model: nn.Module, phase: str = "train"):
        """Call at the start of each training step (before forward pass)."""
        self._step_count += 1
        if not self._should_log(step):
            return
        self._snapshot(step, f"{phase}_step_start", model=model)

    def on_forward_end(
        self,
        step: int,
        noisy_input: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
        model: Optional[nn.Module] = None,
        cap_len: int = 256,
    ):
        """Call after the DiT forward pass with the key tensors."""
        if not self._should_log(step):
            return
        self._snapshot(
            step, "forward_end", model=model,
            noisy_input=noisy_input, model_pred=model_pred,
            timesteps=timesteps, cap_len=cap_len
        )

    def on_phase_boundary(self, step: int, phase: str, model: Optional[nn.Module] = None):
        """
        Call at critical phase boundaries regardless of log_every_n_steps.
        phase examples: "sampling_start", "sampling_end", "checkpoint_save",
                        "text_encoder_cache_start", "text_encoder_cache_end"
        """
        if not self.cfg.enabled:
            return
        if not self.cfg.log_on_phase_change:
            return
        self._snapshot(step, phase, model=model)

    def on_backward_end(self, step: int, model: Optional[nn.Module] = None):
        """Call after accelerator.backward() to check if block devices restored correctly."""
        if not self._should_log(step):
            return
        self._snapshot(step, "backward_end", model=model)

    def report_block_swap_state(self, step: int, model: nn.Module):
        """
        Dedicated check: print the blocks_to_swap attribute and offloader state.
        Useful for confirming block swap enable/disable around sampling.
        """
        if not self.cfg.enabled:
            return
        lines = [f"\n[LuminaDebug] step={step}  BLOCK SWAP STATE"]
        try:
            bts = getattr(model, "blocks_to_swap", "attr_missing")
            lines.append(f"  blocks_to_swap={bts}")
            offloader = getattr(model, "offloader_main", None)
            if offloader is not None:
                lines.append(f"  offloader_main type: {type(offloader).__name__}")
                bs = getattr(offloader, "blocks_to_swap", "?")
                num_blocks = getattr(offloader, "num_blocks", "?")
                lines.append(f"  offloader.blocks_to_swap={bs}  offloader.num_blocks={num_blocks}")
            else:
                lines.append("  offloader_main: None (block swap not initialized)")
        except Exception as e:
            lines.append(f"  [error reading offloader state: {e}]")
        self._emit("\n".join(lines))

    def report_lora_norms(self, step: int, network: nn.Module):
        """Report LoRA weight norms to check for training instability."""
        if not self.cfg.enabled or not self.cfg.log_lora_norms:
            return
        lines = [f"\n[LuminaDebug] step={step}  LORA NORMS"]
        try:
            norms = []
            for name, param in network.named_parameters():
                if param.requires_grad and ("lora_up" in name or "lora_down" in name):
                    with torch.no_grad():
                        norms.append((name, param.norm().item()))
            if norms:
                norms.sort(key=lambda x: x[1], reverse=True)
                for name, norm in norms[:10]:
                    lines.append(f"  {name}: {norm:.6f}")
                if len(norms) > 10:
                    lines.append(f"  ... ({len(norms)} total LoRA params)")
            else:
                lines.append("  No LoRA params with requires_grad found")
        except Exception as e:
            lines.append(f"  [error: {e}]")
        self._emit("\n".join(lines))

    def close(self):
        if self._file_handler:
            try:
                self._file_handler.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: one-shot VRAM dump (callable from anywhere)
# ─────────────────────────────────────────────────────────────────────────────

def dump_vram(label: str = ""):
    """Quick standalone VRAM dump — call from anywhere during debugging."""
    if not torch.cuda.is_available():
        logger.info(f"[dump_vram] {label}: CUDA not available")
        return
    snap = VRAMSnapshot.capture()
    logger.info(f"[dump_vram] {label}: {snap.format()}")


def dump_block_devices(model: nn.Module, label: str = ""):
    """Quick standalone block device dump — call from anywhere during debugging."""
    snap = _block_device_summary(model)
    logger.info(f"[dump_block_devices] {label}:\n{_format_block_devices(snap)}")
