from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class AutomaskSettings:
    enabled: bool = False
    alpha: int = 128
    shrink: int = 1
    blur: float = 3.0
    model: str = "base-nightly"

    def normalized(self) -> "AutomaskSettings":
        return AutomaskSettings(
            enabled=bool(self.enabled),
            alpha=max(1, min(255, int(self.alpha))),
            shrink=max(0, int(self.shrink)),
            blur=max(0.0, float(self.blur)),
            model=str(self.model or "base-nightly").strip() or "base-nightly",
        )

    def to_metadata(self) -> dict[str, Any]:
        value = self.normalized()
        return {
            "automask_enabled": bool(value.enabled),
            "automask_alpha": int(value.alpha),
            "automask_shrink": int(value.shrink),
            "automask_blur": float(value.blur),
            "automask_model": value.model,
        }


def fill_transparent_rgb_with_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8, copy=True)
    transparent = arr[:, :, 3] == 0
    arr[transparent, 0:3] = 255
    return Image.fromarray(arr, mode="RGBA")


def postprocess_mask(mask: Image.Image, settings: AutomaskSettings) -> Image.Image:
    value = settings.normalized()
    processed = mask.convert("L")
    if value.shrink > 0:
        processed = processed.filter(ImageFilter.MinFilter(value.shrink * 2 + 1))
    if value.blur > 0:
        processed = processed.filter(ImageFilter.GaussianBlur(value.blur))
    if value.alpha > 0:
        processed = processed.point(lambda pixel: max(pixel, value.alpha))
    return processed


def generate_automask_alpha(
    image: Image.Image,
    *,
    remover: Any = None,
    settings: AutomaskSettings | None = None,
) -> Image.Image:
    value = (settings or AutomaskSettings()).normalized()
    clean_rgba = fill_transparent_rgb_with_white(image)
    rgb_for_model = clean_rgba.convert("RGB")

    if remover is None:
        from transparent_background import Remover

        remover = Remover(mode=value.model)

    mask = remover.process(rgb_for_model, type="map")
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask))
    if mask.size != rgb_for_model.size:
        mask = mask.resize(rgb_for_model.size, Image.Resampling.BILINEAR)
    return postprocess_mask(mask, value)


def alpha_mask_to_uint8(mask: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    elif torch is not None and isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.rint(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def alpha_mask_from_uint8(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def metadata_matches(metadata: Mapping[str, Any], settings: AutomaskSettings) -> bool:
    value = settings.normalized()
    expected = value.to_metadata()
    for key, expected_value in expected.items():
        actual = _metadata_value(metadata, key)
        if isinstance(expected_value, float):
            try:
                if abs(float(actual) - expected_value) > 1e-6:
                    return False
            except Exception:
                return False
        else:
            if actual != expected_value:
                return False
    return True
