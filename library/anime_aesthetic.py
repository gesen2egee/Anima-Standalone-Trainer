"""Minimal anime DBAesthetic percentile runtime used during latent caching."""

import gc
import logging
import os
from threading import Lock
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

_REPO_ID = "deepghs/anime_aesthetic"
_MODEL_NAME = "swinv2pv3_v0_448_ls0.2_x"
_MODEL_FILE = f"{_MODEL_NAME}/model.onnx"
_SAMPLES_FILE = f"{_MODEL_NAME}/samples.npz"
_QUALITY_BY_MODEL_OUTPUT = np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)

_session = None
_percentile_samples = None
_lock = Lock()


def _get_providers(ort) -> list[str]:
    requested = os.environ.get("AES_ONNX_PROVIDER", os.environ.get("ONNX_MODE", "")).strip().lower()
    available = set(ort.get_available_providers())
    if requested in {"cpu", "cpuexecutionprovider"}:
        return ["CPUExecutionProvider"]
    if requested in {"cuda", "gpu", "cudaexecutionprovider"}:
        if "CUDAExecutionProvider" not in available:
            logger.warning("AES CUDA provider is unavailable; falling back to CPU")
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_session():
    global _session
    with _lock:
        if _session is None:
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError("AES scoring requires onnxruntime or onnxruntime-gpu") from e

            model_path = hf_hub_download(_REPO_ID, _MODEL_FILE)
            providers = _get_providers(ort)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if providers[0] == "CPUExecutionProvider":
                options.intra_op_num_threads = os.cpu_count() or 1
            try:
                _session = ort.InferenceSession(model_path, options, providers=providers)
            except Exception:
                if providers[0] == "CPUExecutionProvider":
                    raise
                logger.exception("AES CUDA session failed; retrying on CPU")
                _session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
            logger.info("loaded anime DBAesthetic model with providers: %s", _session.get_providers())
        return _session


def _get_percentile_samples() -> tuple[np.ndarray, np.ndarray]:
    global _percentile_samples
    with _lock:
        if _percentile_samples is None:
            samples_path = hf_hub_download(_REPO_ID, _SAMPLES_FILE)
            with np.load(samples_path) as data:
                samples = np.asarray(data["arr_0"])
            _percentile_samples = (samples[0], samples[1])
        return _percentile_samples


def _prepare_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as source:
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            image = source.convert("RGB")
        image = image.resize((448, 448), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = (array - 0.5) / 0.5
    return np.ascontiguousarray(array[None, ...], dtype=np.float32)


def anime_dbaesthetic(image_path: str) -> float:
    """Return the same 0-1 percentile used by imgutils anime_dbaesthetic."""
    output = _get_session().run(["output"], {"input": _prepare_image(image_path)})[0][0]
    score = float(np.dot(np.asarray(output, dtype=np.float32), _QUALITY_BY_MODEL_OUTPUT))
    x, y = _get_percentile_samples()
    clipped = float(np.clip(score, x.min(), x.max()))
    index = int(np.searchsorted(x, clipped))
    if index >= x.shape[0] - 1:
        return float(y[-1])
    x0, x1 = float(x[index]), float(x[index + 1])
    y0, y1 = float(y[index]), float(y[index + 1])
    if np.isclose(x1, x0):
        return y0
    return float(np.clip((clipped - x0) / (x1 - x0) * (y1 - y0) + y0, y.min(), y.max()))


def release_anime_dbaesthetic(device: Optional[torch.device] = None) -> None:
    """Drop the ONNX session and percentile arrays after latent caching."""
    global _session, _percentile_samples
    with _lock:
        session = _session
        _session = None
        _percentile_samples = None
    del session
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("released anime DBAesthetic ONNX session after latent caching")
