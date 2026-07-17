"""KonIQ NR-IQA quality runtime used by TQA latent caching."""

import gc
import logging
import os
from threading import RLock
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

_BASE_MODEL = "google/siglip2-so400m-patch16-512"
_BASE_REVISION = "ceea1cba8130d8271436da4828633198c176a775"
_ADAPTER_MODEL = "mlx-community/SigLIP2-NR-IQA-KonIQ"
_ADAPTER_REVISION = "8349820ad6cfbc49183058cc124d75b25a167eeb"
TQA_QUALITY_CACHE_MODEL = (
    f"{_ADAPTER_MODEL}@{_ADAPTER_REVISION}+{_BASE_MODEL}@{_BASE_REVISION}:pytorch-merged-v1"
)

_backbone = None
_head = None
_device: Optional[torch.device] = None
_dtype: Optional[torch.dtype] = None
_lock = RLock()


class _NRIQAHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(1152, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(self.fc1(features))
        hidden = F.leaky_relu(self.fc2(hidden), negative_slope=0.05)
        return self.fc3(hidden).squeeze(-1)


def _runtime() -> tuple[torch.device, torch.dtype]:
    requested = os.environ.get("TQA_DEVICE", "").strip().lower()
    use_cuda = torch.cuda.is_available() and requested != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, torch.float16 if device.type == "cuda" else torch.float32


def _ensure_loaded() -> None:
    global _backbone, _head, _device, _dtype
    with _lock:
        if _backbone is not None and _head is not None:
            return
        try:
            from transformers import SiglipVisionModel
        except ImportError as e:
            raise ImportError("TQA KonIQ quality scoring requires transformers") from e

        _device, _dtype = _runtime()
        try:
            adapter_path = hf_hub_download(
                _ADAPTER_MODEL,
                "adapter.safetensors",
                revision=_ADAPTER_REVISION,
            )
            adapter = load_file(adapter_path, device="cpu")
            backbone = SiglipVisionModel.from_pretrained(
                _BASE_MODEL,
                revision=_BASE_REVISION,
                dtype=torch.float32,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            with torch.no_grad():
                for index, layer in enumerate(backbone.vision_model.encoder.layers):
                    for projection_name in ("q_proj", "k_proj"):
                        projection = getattr(layer.self_attn, projection_name)
                        prefix = f"backbone.layers.{index}.self_attn.{projection_name}"
                        lora_a = adapter[f"{prefix}.lora_a"]
                        lora_b = adapter[f"{prefix}.lora_b"]
                        projection.weight.add_((lora_b @ lora_a) * 2.0)

            head = _NRIQAHead()
            head.load_state_dict(
                {
                    key.removeprefix("head."): value
                    for key, value in adapter.items()
                    if key.startswith("head.")
                }
            )
            _backbone = backbone.eval().requires_grad_(False).to(device=_device, dtype=_dtype)
            _head = head.eval().requires_grad_(False).to(device=_device, dtype=_dtype)
            logger.info(
                "loaded TQA KonIQ NR-IQA model on %s with dtype %s",
                _device,
                _dtype,
            )
        except Exception:
            release_tqa_models(_device)
            raise


def _prepare_image(image_path: str) -> torch.Tensor:
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=_device, dtype=_dtype)


def score_tqa_quality(image_path: str) -> float:
    """Return KonIQ MOS quality (~0-100, higher is better)."""
    _ensure_loaded()
    with torch.inference_mode():
        features = _backbone(pixel_values=_prepare_image(image_path)).pooler_output
        score = (_head(features).float() * 100.0).reshape(-1)[0].item()
    return float(score)


def release_tqa_models(device: Optional[torch.device] = None) -> None:
    """Release the KonIQ backbone and IQA head after latent caching."""
    global _backbone, _head, _device, _dtype
    with _lock:
        backbone = _backbone
        head = _head
        runtime_device = _device
        _backbone = None
        _head = None
        _device = None
        _dtype = None
    del backbone, head
    gc.collect()
    cleanup_device = device or runtime_device
    if cleanup_device is not None and cleanup_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("released TQA KonIQ NR-IQA model after latent caching")
