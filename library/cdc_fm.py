"""Carré du champ flow-matching geometry for latent image trainers.

The cache stores a low-rank eigendecomposition of the local covariance for
each training image.  Training keeps the standard Gaussian source at
``sigma=1`` and applies the geometric covariance at the data endpoint
``sigma=0``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from library import strategy_base


logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def _stable_key(value: str) -> str:
    return os.path.normcase(os.path.abspath(str(value)))


@dataclass(frozen=True)
class _LatentRecord:
    key: str
    npz_path: str
    bucket_reso: Tuple[int, int]


def _iter_datasets(dataset_group) -> Iterable:
    datasets = getattr(dataset_group, "datasets", None)
    return datasets if datasets is not None else [dataset_group]


def _collect_records(dataset_group) -> List[_LatentRecord]:
    records: Dict[str, _LatentRecord] = {}
    for dataset in _iter_datasets(dataset_group):
        for info in getattr(dataset, "image_data", {}).values():
            if not info.latents_npz:
                raise ValueError(
                    "CDC-FM requires disk-cached latents for every training image. "
                    "Enable --cache_latents and --cache_latents_to_disk."
                )
            key = _stable_key(info.absolute_path)
            record = _LatentRecord(key, os.path.abspath(info.latents_npz), tuple(info.bucket_reso))
            existing = records.get(key)
            if existing is not None and existing != record:
                raise ValueError(f"CDC-FM found conflicting latent caches for {key}")
            records[key] = record
    if not records:
        raise ValueError("CDC-FM could not find any training latents")
    return sorted(records.values(), key=lambda item: item.key)


def _normalise_latent(latent: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(latent), dtype=torch.float32)
    if tensor.ndim == 4:
        if tensor.shape[1] != 1:
            raise ValueError(f"CDC-FM only supports single-frame image latents, got {tuple(tensor.shape)}")
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"CDC-FM expected a [C,H,W] latent, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _configuration_hash(records: Sequence[_LatentRecord], settings: Dict) -> str:
    files = []
    for record in records:
        stat = os.stat(record.npz_path)
        files.append((record.key, record.npz_path, record.bucket_reso, stat.st_size, stat.st_mtime_ns))
    payload = json.dumps({"version": CACHE_VERSION, "settings": settings, "files": files}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _cache_filename(key: str, shape: Sequence[int]) -> str:
    digest = hashlib.sha256(f"{key}|{tuple(shape)}".encode("utf-8")).hexdigest()[:24]
    return f"{digest}.safetensors"


def _knn_graph(data: torch.Tensor, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact k-NN distances/indices, including self at column zero."""
    n = data.shape[0]
    k = min(max(2, int(k)), n)

    def compute(target_device: torch.device):
        database = data.to(target_device, dtype=torch.float32)
        distance_rows = []
        index_rows = []
        chunk_size = min(128, n)
        for start in range(0, n, chunk_size):
            query = database[start : start + chunk_size]
            distances = torch.cdist(query, database)
            values, indices = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
            distance_rows.append(values.cpu())
            index_rows.append(indices.cpu())
        return torch.cat(distance_rows), torch.cat(index_rows)

    try:
        return compute(device)
    except torch.OutOfMemoryError:
        if device.type != "cuda":
            raise
        logger.warning("CDC-FM k-NN exceeded GPU memory; retrying on CPU")
        torch.cuda.empty_cache()
        return compute(torch.device("cpu"))


def _estimate_bucket_geometry(
    data: torch.Tensor,
    *,
    k_neighbors: int,
    k_bandwidth: int,
    cdc_dim: int,
    gamma: float,
    bandwidth_rescale: float,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Estimate low-rank local covariance square-root factors for one bucket."""
    distances, indices = _knn_graph(data, k_neighbors, device)
    n, k = indices.shape
    bandwidth_index = min(max(1, int(k_bandwidth)), k - 1)
    bandwidths = distances[:, bandwidth_index].clamp_min(1e-10)
    local_limits = distances[:, 1].square() / 9.0
    global_limit = torch.quantile(local_limits, 0.9)

    work_device = device
    try:
        work_data = data.to(work_device, dtype=torch.float32)
    except torch.OutOfMemoryError:
        logger.warning("CDC-FM covariance data exceeded GPU memory; retrying on CPU")
        torch.cuda.empty_cache()
        work_device = torch.device("cpu")
        work_data = data.float()

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for point_index in range(n):
        neighbour_indices = indices[point_index].to(work_device)
        neighbours = work_data.index_select(0, neighbour_indices)
        neighbour_distances = distances[point_index].to(work_device)
        neighbour_bandwidths = bandwidths.index_select(0, indices[point_index]).to(work_device)
        epsilon = (bandwidth_rescale * bandwidths[point_index].to(work_device) * neighbour_bandwidths).clamp_min(1e-10)
        weights = torch.exp(-neighbour_distances.square() / epsilon)
        weights = weights / weights.sum().clamp_min(1e-20)

        centre = torch.sum(weights[:, None] * neighbours, dim=0)
        weighted = torch.sqrt(weights)[:, None] * (neighbours - centre)
        gram = weighted @ weighted.T
        eigenvalues_small, eigenvectors_small = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues_small, descending=True)
        rank = min(int(cdc_dim), k, int(torch.count_nonzero(eigenvalues_small > 1e-12).item()))

        if rank == 0:
            vectors = torch.zeros((cdc_dim, data.shape[1]), dtype=torch.float16)
            values = torch.zeros(cdc_dim, dtype=torch.float32)
            results.append((vectors, values))
            continue

        selected = order[:rank]
        raw_values = eigenvalues_small[selected].clamp_min(1e-12)
        left_vectors = eigenvectors_small[:, selected]
        vectors = (left_vectors.T @ weighted) / torch.sqrt(raw_values)[:, None]

        # Paper Appendix E, Eq. 33: constrain the largest local eigenvalue by
        # both the nearest-neighbour gap and the global 90th percentile cap.
        allowed_max = torch.minimum(local_limits[point_index], global_limit).to(work_device)
        scale = allowed_max / raw_values[0]
        values = raw_values * scale * gamma

        if rank < cdc_dim:
            vectors = torch.nn.functional.pad(vectors, (0, 0, 0, cdc_dim - rank))
            values = torch.nn.functional.pad(values, (0, cdc_dim - rank))

        results.append((vectors.cpu().half().contiguous(), values.cpu().float().contiguous()))
    return results


def _manifest_is_complete(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != CACHE_VERSION:
            return False
        root = manifest_path.parent
        return all(entry.get("fallback") or (root / entry["artifact"]).exists() for entry in manifest["entries"].values())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_cdc_cache(args, dataset_group, accelerator, architecture: str) -> "CDCGeometryCache":
    records = _collect_records(dataset_group)
    settings = {
        "architecture": architecture,
        "k_neighbors": int(args.cdc_k_neighbors),
        "k_bandwidth": int(args.cdc_k_bandwidth),
        "cdc_dim": int(args.cdc_dim),
        "gamma": float(args.cdc_gamma),
        "bandwidth_rescale": float(args.cdc_bandwidth_rescale),
        "min_bucket_size": int(args.cdc_min_bucket_size),
        "immiscible_image_scale": float(getattr(args, "immiscible_image_scale", 1.0)),
    }
    config_hash = _configuration_hash(records, settings)
    cache_root = Path(args.cdc_cache_dir).expanduser().resolve() / architecture / config_hash
    manifest_path = cache_root / "manifest.json"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and (args.cdc_force_recache or not _manifest_is_complete(manifest_path)):
        cache_root.mkdir(parents=True, exist_ok=True)
        strategy = strategy_base.LatentsCachingStrategy.get_strategy()
        if strategy is None:
            raise RuntimeError("CDC-FM latent caching strategy is not initialized")

        grouped: Dict[Tuple[int, ...], List[Tuple[_LatentRecord, torch.Tensor]]] = defaultdict(list)
        for record in records:
            latent, *_ = strategy.load_latents_from_disk(record.npz_path, record.bucket_reso)
            tensor = _normalise_latent(latent) * settings["immiscible_image_scale"]
            grouped[tuple(tensor.shape)].append((record, tensor))

        manifest = {"version": CACHE_VERSION, "hash": config_hash, "settings": settings, "entries": {}}
        total_buckets = len(grouped)
        for bucket_number, (shape, bucket) in enumerate(sorted(grouped.items()), start=1):
            logger.info("CDC-FM bucket %d/%d: shape=%s samples=%d", bucket_number, total_buckets, shape, len(bucket))
            if len(bucket) < args.cdc_min_bucket_size:
                logger.warning(
                    "CDC-FM bucket %s has only %d samples (< %d); using Gaussian fallback",
                    shape,
                    len(bucket),
                    args.cdc_min_bucket_size,
                )
                for record, _ in bucket:
                    manifest["entries"][record.key] = {"shape": list(shape), "fallback": True}
                continue

            flattened = torch.stack([latent.reshape(-1) for _, latent in bucket])
            geometry = _estimate_bucket_geometry(
                flattened,
                k_neighbors=min(args.cdc_k_neighbors, len(bucket)),
                k_bandwidth=args.cdc_k_bandwidth,
                cdc_dim=args.cdc_dim,
                gamma=args.cdc_gamma,
                bandwidth_rescale=args.cdc_bandwidth_rescale,
                device=accelerator.device,
            )
            for (record, _), (vectors, values) in zip(bucket, geometry):
                filename = _cache_filename(record.key, shape)
                save_file({"eigenvectors": vectors, "eigenvalues": values}, str(cache_root / filename))
                manifest["entries"][record.key] = {
                    "shape": list(shape),
                    "fallback": False,
                    "artifact": filename,
                }

        temp_manifest = manifest_path.with_suffix(".json.tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        logger.info("CDC-FM cache ready: %s", manifest_path)

    accelerator.wait_for_everyone()
    if not _manifest_is_complete(manifest_path):
        raise RuntimeError(f"CDC-FM cache is incomplete: {manifest_path}")
    return CDCGeometryCache(manifest_path, max_memory_entries=args.cdc_cache_memory_entries)


class CDCGeometryCache:
    def __init__(self, manifest_path: Path | str, max_memory_entries: int = 32):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = {_stable_key(key): value for key, value in self.manifest["entries"].items()}
        self.max_memory_entries = max(1, int(max_memory_entries))
        self._memory: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._warned_missing = set()

    def _load(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]]:
        stable = _stable_key(key)
        entry = self.entries.get(stable)
        if entry is None or entry.get("fallback"):
            if entry is None and stable not in self._warned_missing:
                logger.warning("CDC-FM geometry is missing for %s; using Gaussian fallback", stable)
                self._warned_missing.add(stable)
            return None
        artifact = entry["artifact"]
        cached = self._memory.get(artifact)
        if cached is None:
            tensors = load_file(str(self.root / artifact), device="cpu")
            cached = (tensors["eigenvectors"], tensors["eigenvalues"])
            self._memory[artifact] = cached
            if len(self._memory) > self.max_memory_entries:
                self._memory.popitem(last=False)
        else:
            self._memory.move_to_end(artifact)
        return cached[0], cached[1], tuple(entry["shape"])

    def correction(
        self,
        noise: torch.Tensor,
        keys: Sequence[str],
        flippeds: Optional[Sequence[bool]] = None,
    ) -> torch.Tensor:
        if len(keys) != noise.shape[0]:
            raise ValueError(f"CDC-FM key count {len(keys)} does not match batch size {noise.shape[0]}")
        flips = list(flippeds) if flippeds is not None else [False] * len(keys)
        # Keep the low-rank square-root projection in fp32.  The smallest
        # eigenvalues easily underflow in bf16/fp16 during normal training.
        flattened_noise = noise.float().reshape(noise.shape[0], -1)
        corrections = []
        for index, (key, flipped) in enumerate(zip(keys, flips)):
            loaded = self._load(key)
            if loaded is None:
                corrections.append(torch.zeros_like(flattened_noise[index]))
                continue
            vectors, values, shape = loaded
            if tuple(noise.shape[1:]) != shape:
                raise ValueError(
                    f"CDC-FM latent shape mismatch for {key}: cache={shape}, batch={tuple(noise.shape[1:])}"
                )
            vectors = vectors.to(device=noise.device, dtype=torch.float32)
            values = values.to(device=noise.device, dtype=torch.float32)
            if flipped:
                vectors = torch.flip(vectors.reshape(vectors.shape[0], *shape), dims=[-1]).reshape_as(vectors)
            projection = vectors @ flattened_noise[index]
            correction = vectors.T @ (torch.sqrt(values.clamp_min(0)) * projection)
            corrections.append(correction)
        return torch.stack(corrections).reshape_as(noise).to(dtype=noise.dtype)


def apply_cdc_flow_path(
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    correction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the CDC probability-path sample and its exact velocity target.

    This project uses ``sigma=0`` for data and ``sigma=1`` for Gaussian noise,
    the reverse orientation of Algorithm 2 in the paper.
    """
    path_sample = (1.0 - sigmas) * latents + sigmas * noise + (1.0 - sigmas) * correction
    target = noise - latents - correction
    return path_sample, target
