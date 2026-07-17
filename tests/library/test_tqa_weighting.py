from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library import train_util
from library import anime_aesthetic, tqa_metrics
from library.tqa_metrics import TQA_QUALITY_CACHE_MODEL
from library.flux_train_utils import get_noisy_model_input_and_timesteps
from library.strategy_base import LatentsCachingStrategy


class MockNoiseScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)


def test_weighted_percentiles_respect_repeats_and_ties():
    values = [1.0, 2.0, 2.0, 3.0]
    weights = [1, 2, 1, 2]

    result = train_util.DatasetGroup._weighted_percentile_ranks(values, weights)

    assert result == pytest.approx([0.0, 0.4, 0.4, 0.9])


def test_dataset_tqa_percentiles_map_difference_extremes_to_exact_shift_range(monkeypatch):
    infos = [
        SimpleNamespace(
            tqa_koniq_quality_score=quality,
            tqa_dbaes_score=aesthetic,
            tqa_quality_percentile=None,
            tqa_dbaes_percentile=None,
            tqa_shift=None,
            latents_npz=None,
            absolute_path=f"image-{index}.png",
            num_repeats=2,
        )
        for index, (quality, aesthetic) in enumerate([(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)])
    ]
    strategy = SimpleNamespace(cache_tqa_scores=True)
    monkeypatch.setattr(
        LatentsCachingStrategy,
        "get_strategy",
        classmethod(lambda cls: strategy),
    )
    group = train_util.DatasetGroup.__new__(train_util.DatasetGroup)
    group.datasets = [SimpleNamespace(image_data={str(i): info for i, info in enumerate(infos)})]

    group._normalize_tqa_scores()

    assert [info.tqa_shift for info in infos] == pytest.approx([1.5, 1.0, 0.5])


def test_tqa_loss_interpolates_quality_to_aesthetic_by_timestep():
    args = SimpleNamespace(
        aes_loss_weighting=False,
        tqa_loss_weighting=True,
        tqa_loss_weighting_schedule=False,
        max_train_steps=100,
    )
    batch = {
        "tqa_quality_percentiles": torch.tensor([0.25, 0.75]),
        "tqa_dbaes_percentiles": torch.tensor([0.75, 0.25]),
    }
    loss = torch.ones(2)

    low = train_util.apply_aes_loss_weighting(loss, batch, args, 0, sigmas=torch.zeros(2, 1, 1, 1))
    high = train_util.apply_aes_loss_weighting(loss, batch, args, 0, sigmas=torch.ones(2, 1, 1, 1))
    middle = train_util.apply_aes_loss_weighting(
        loss, batch, args, 0, sigmas=torch.full((2, 1, 1, 1), 0.5)
    )

    assert torch.allclose(low, torch.tensor([0.5, 1.5]))
    assert torch.allclose(high, torch.tensor([1.5, 0.5]))
    assert torch.allclose(middle, torch.ones(2))


def test_tqa_and_dbaes_use_geometric_mean():
    args = SimpleNamespace(
        aes_loss_weighting=True,
        aes_loss_weighting_schedule=False,
        tqa_loss_weighting=True,
        tqa_loss_weighting_schedule=False,
        max_train_steps=100,
    )
    batch = {
        "aes_scores": torch.tensor([2.0]),
        "tqa_quality_percentiles": torch.tensor([0.125]),
        "tqa_dbaes_percentiles": torch.tensor([0.125]),
    }

    weighted = train_util.apply_aes_loss_weighting(
        torch.ones(1), batch, args, 0, sigmas=torch.zeros(1, 1, 1, 1)
    )

    assert torch.allclose(weighted, torch.tensor([np.sqrt(0.5)], dtype=torch.float32))


def test_progressive_tqa_weighting_transitions_from_one_to_timestep_weight():
    args = SimpleNamespace(
        aes_loss_weighting=False,
        tqa_loss_weighting=True,
        tqa_loss_weighting_schedule=True,
        max_train_steps=101,
    )
    batch = {
        "tqa_quality_percentiles": torch.tensor([0.25]),
        "tqa_dbaes_percentiles": torch.tensor([0.75]),
    }
    sigma = torch.ones(1, 1, 1, 1)

    start = train_util.apply_aes_loss_weighting(torch.ones(1), batch, args, 0, sigmas=sigma)
    middle = train_util.apply_aes_loss_weighting(torch.ones(1), batch, args, 50, sigmas=sigma)
    end = train_util.apply_aes_loss_weighting(torch.ones(1), batch, args, 100, sigmas=sigma)

    assert torch.allclose(start, torch.tensor([1.0]))
    assert torch.allclose(middle, torch.tensor([1.25]))
    assert torch.allclose(end, torch.tensor([1.5]))


def test_autoshift_tqa_uses_cached_half_to_one_and_half_range(monkeypatch):
    args = SimpleNamespace(
        timestep_sampling="autoshift_tqa",
        sigmoid_scale=1.0,
        discrete_flow_shift=9.0,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
    )
    latents = torch.zeros(3, 1, 2, 2)
    noise = torch.ones_like(latents)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.zeros(3))

    _, timesteps, _ = get_noisy_model_input_and_timesteps(
        args,
        MockNoiseScheduler(),
        latents,
        noise,
        torch.device("cpu"),
        torch.float32,
        tqa_shift_values=[0.5, 1.0, 1.5],
    )

    assert torch.allclose(timesteps, torch.tensor([1000 / 3, 500.0, 600.0]), atol=1e-4)


def test_tqa_missing_npz_fields_forces_recache_even_when_checks_are_skipped(tmp_path):
    cache = tmp_path / "latent.npz"
    strategy = LatentsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
        cache_tqa_scores=True,
    )
    np.savez(
        cache,
        latents=np.zeros((1, 4, 4), dtype=np.float32),
        tqa_quality_score=np.array(4.0, dtype=np.float32),
        tqa_aesthetic_score=np.array(3.0, dtype=np.float32),
    )

    assert not strategy._default_is_disk_cached_latents_expected(
        8, (32, 32), str(cache), False, False
    )

    np.savez(
        cache,
        latents=np.zeros((1, 4, 4), dtype=np.float32),
        tqa_koniq_quality_score=np.array(70.0, dtype=np.float32),
        tqa_dbaes_score=np.array(0.75, dtype=np.float32),
        tqa_quality_model=np.array("different-model"),
    )
    assert not strategy._default_is_disk_cached_latents_expected(
        8, (32, 32), str(cache), False, False
    )

    np.savez(
        cache,
        latents=np.zeros((1, 4, 4), dtype=np.float32),
        tqa_koniq_quality_score=np.array(70.0, dtype=np.float32),
        tqa_dbaes_score=np.array(0.75, dtype=np.float32),
        tqa_quality_model=np.array(TQA_QUALITY_CACHE_MODEL),
    )
    assert strategy._default_is_disk_cached_latents_expected(
        8, (32, 32), str(cache), False, False
    )


def test_tqa_and_aes_share_one_dbaes_inference_per_image(monkeypatch):
    infos = [SimpleNamespace(absolute_path=f"image-{index}.png") for index in range(2)]
    calls = {"dbaes": 0, "quality": 0}

    def fake_dbaes(_path):
        calls["dbaes"] += 1
        return 0.25 * calls["dbaes"]

    def fake_quality(_path):
        calls["quality"] += 1
        return 4.0 + calls["quality"]

    monkeypatch.setattr(anime_aesthetic, "anime_dbaesthetic", fake_dbaes)
    monkeypatch.setattr(tqa_metrics, "score_tqa_quality", fake_quality)
    monkeypatch.setattr(
        train_util,
        "load_images_and_masks_for_caching",
        lambda *_args: (
            torch.zeros(2, 3, 8, 8),
            [None, None],
            [(8, 8), (8, 8)],
            [(0, 0, 8, 8), (0, 0, 8, 8)],
        ),
    )
    strategy = LatentsCachingStrategy(
        cache_to_disk=False,
        batch_size=2,
        skip_disk_cache_validity_check=False,
        cache_aes_score=True,
        cache_tqa_scores=True,
    )

    strategy._default_cache_batch_latents(
        lambda images: torch.zeros(images.shape[0], 1, 2, 2),
        torch.device("cpu"),
        torch.float32,
        infos,
        False,
        False,
        False,
    )

    assert calls == {"dbaes": 2, "quality": 2}
    assert [info.aes_score for info in infos] == [0.25, 0.5]
    assert [info.tqa_dbaes_score for info in infos] == [0.25, 0.5]
    assert [info.tqa_koniq_quality_score for info in infos] == [5.0, 6.0]


def test_tqa_quality_cache_identifies_koniq_and_retires_old_siglip_model():
    assert "SigLIP2-NR-IQA-KonIQ" in TQA_QUALITY_CACHE_MODEL
    assert "trojblue" not in TQA_QUALITY_CACHE_MODEL


def test_tqa_save_writes_koniq_field_and_model_identity(tmp_path):
    cache = tmp_path / "latent.npz"
    strategy = LatentsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=False,
        cache_tqa_scores=True,
    )

    strategy.save_latents_to_disk(
        str(cache),
        np.zeros((1, 4, 4), dtype=np.float32),
        (32, 32),
        (0, 0, 32, 32),
        tqa_koniq_quality_score=72.5,
        tqa_dbaes_score=0.75,
    )

    with np.load(cache) as data:
        assert "tqa_quality_score" not in data
        assert float(data["tqa_koniq_quality_score"]) == pytest.approx(72.5)
        assert str(data["tqa_quality_model"].item()) == TQA_QUALITY_CACHE_MODEL
