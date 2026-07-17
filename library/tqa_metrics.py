"""SigLIP quality/aesthetic regression runtime for TQA latent caching."""

import gc
import logging
import os
from threading import RLock
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_IDS = {
    "quality": "trojblue/distill-q-align-quality-siglip2-base",
    "aesthetic": "trojblue/distill-q-align-aesthetic-siglip2-base",
}
_models = {}
_processors = {}
_device: Optional[torch.device] = None
_dtype: Optional[torch.dtype] = None
_lock = RLock()


def _runtime() -> tuple[torch.device, torch.dtype]:
    requested = os.environ.get("TQA_DEVICE", "").strip().lower()
    use_cuda = torch.cuda.is_available() and requested != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    # Percentile order is sensitive to BF16 score quantization, so keep the
    # small regression heads and SigLIP backbones in FP32 while caching.
    return device, torch.float32


def _ensure_loaded() -> None:
    global _device, _dtype
    with _lock:
        if len(_models) == len(_MODEL_IDS):
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as e:
            raise ImportError("TQA scoring requires transformers") from e

        _device, _dtype = _runtime()
        try:
            for name, model_id in _MODEL_IDS.items():
                if name in _models:
                    continue
                processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
                model = AutoModelForImageClassification.from_pretrained(
                    model_id,
                    dtype=_dtype,
                )
                model.eval().requires_grad_(False).to(_device)
                _processors[name] = processor
                _models[name] = model
            logger.info(
                "loaded TQA SigLIP models on %s with dtype %s",
                _device,
                _dtype,
            )
        except Exception:
            release_tqa_models(_device)
            raise


def score_tqa(image_path: str) -> tuple[float, float]:
    """Return raw (quality, aesthetic) regression scores for one image."""
    _ensure_loaded()
    with Image.open(image_path) as source:
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = source.convert("RGB")

    scores = []
    with torch.inference_mode():
        for name in ("quality", "aesthetic"):
            inputs = _processors[name](images=image, return_tensors="pt")
            inputs = {
                key: value.to(device=_device, dtype=_dtype if value.is_floating_point() else None)
                for key, value in inputs.items()
            }
            score = _models[name](**inputs).logits.float().reshape(-1)[0].item()
            scores.append(float(score))
    return scores[0], scores[1]


def release_tqa_models(device: Optional[torch.device] = None) -> None:
    """Release both SigLIP models after the complete latent-cache pass."""
    global _device, _dtype
    with _lock:
        models = list(_models.values())
        processors = list(_processors.values())
        _models.clear()
        _processors.clear()
        runtime_device = _device
        _device = None
        _dtype = None
    del models, processors
    gc.collect()
    cleanup_device = device or runtime_device
    if cleanup_device is not None and cleanup_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("released TQA SigLIP models after latent caching")
