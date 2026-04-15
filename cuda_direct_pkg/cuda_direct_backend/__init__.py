"""
cuda_direct — GPU-to-GPU ProcessGroup backend for PyTorch distributed on Windows.

Quick start:
    # Auto-detect and activate (recommended)
    from cuda_direct_backend.auto import activate
    activate()

    # Or just register the backend without NCCL spoofing
    from cuda_direct_backend import register_backend
    register_backend()

    # Or run diagnostics to see what's available
    from cuda_direct_backend.auto import activate
    activate(mode="diagnose")
"""

from .process_group import register_backend, ProcessGroupCudaDirect
from .diagnostics import diagnose
from .comm_timer import CommTimer, CommEvent, dtype_str

# Only auto-register on Windows with 2+ CUDA GPUs.
# On Linux (where NCCL works), or single-GPU, do nothing.
import os as _os

try:
    import torch as _torch
    _should_auto_register = (
        _os.name == "nt"
        and _torch.cuda.is_available()
        and _torch.cuda.device_count() > 1
    )
except Exception:
    _should_auto_register = False

if _should_auto_register:
    try:
        register_backend()
    except Exception:
        pass  # Already registered or torch.distributed not ready yet
