import os
import sys
import argparse
import pytest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.automask import (
    AutomaskSettings,
    alpha_mask_from_uint8,
    alpha_mask_to_uint8,
    fill_transparent_rgb_with_white,
    generate_automask_alpha,
    metadata_matches,
)
from library.strategy_base import LatentsCachingStrategy
from library.train_util import ImageInfo


class FakeRemover:
    def __init__(self, mask: Image.Image):
        self.mask = mask
        self.seen_image = None

    def process(self, image: Image.Image, type: str = "map"):
        self.seen_image = image.copy()
        assert type == "map"
        return self.mask


def test_fill_transparent_rgb_with_white_preserves_visible_pixels():
    image = Image.fromarray(
        np.array(
            [
                [[10, 20, 30, 255], [100, 110, 120, 0]],
                [[40, 50, 60, 128], [70, 80, 90, 0]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    )

    filled = fill_transparent_rgb_with_white(image)
    arr = np.array(filled)

    assert arr[0, 0].tolist() == [10, 20, 30, 255]
    assert arr[1, 0].tolist() == [40, 50, 60, 128]
    assert arr[0, 1].tolist() == [255, 255, 255, 0]
    assert arr[1, 1].tolist() == [255, 255, 255, 0]


def test_generate_automask_alpha_uses_white_filled_rgb_and_default_alpha():
    source = Image.fromarray(
        np.array(
            [
                [[10, 20, 30, 255], [200, 10, 20, 0]],
                [[40, 50, 60, 255], [70, 80, 90, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    )
    fake_mask = Image.fromarray(np.array([[255, 0], [32, 255]], dtype=np.uint8), mode="L")
    remover = FakeRemover(fake_mask)

    alpha = generate_automask_alpha(
        source,
        remover=remover,
        settings=AutomaskSettings(enabled=True, alpha=128, shrink=0, blur=0),
    )

    assert alpha.mode == "L"
    assert np.array(alpha).tolist() == [[255, 128], [128, 255]]
    assert remover.seen_image.mode == "RGB"
    assert np.array(remover.seen_image)[0, 1].tolist() == [255, 255, 255]


def test_alpha_mask_uint8_round_trip_converts_to_training_float_range():
    mask = Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8), mode="L")

    packed = alpha_mask_to_uint8(mask)
    unpacked = alpha_mask_from_uint8(packed)

    assert packed.dtype == np.uint8
    assert np.allclose(unpacked, np.array([[0.0, 128 / 255.0, 1.0]], dtype=np.float32))


def test_metadata_matches_rejects_stale_automask_settings():
    settings = AutomaskSettings(enabled=True, alpha=128, shrink=1, blur=3, model="base-nightly")
    metadata = settings.to_metadata()

    assert metadata_matches(metadata, settings)
    assert not metadata_matches(metadata, AutomaskSettings(enabled=True, alpha=64, shrink=1, blur=3, model="base-nightly"))


def test_metadata_matches_npz_file(tmp_path):
    settings = AutomaskSettings(enabled=True, alpha=128, shrink=1, blur=3, model="base-nightly")
    npz_path = tmp_path / "metadata.npz"
    np.savez(npz_path, **settings.to_metadata())

    with np.load(npz_path) as metadata:
        assert metadata_matches(metadata, settings)


def test_cache_image_loading_uses_automask_alpha(tmp_path):
    from library import train_util

    image_path = tmp_path / "sample.png"
    Image.fromarray(
        np.array(
            [
                [[10, 20, 30, 255], [200, 10, 20, 0]],
                [[40, 50, 60, 255], [70, 80, 90, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGBA",
    ).save(image_path)

    fake_mask = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8), mode="L")
    remover = FakeRemover(fake_mask)
    info = ImageInfo("sample", 1, "", False, str(image_path))
    info.bucket_reso = (2, 2)
    info.resized_size = (2, 2)

    train_util.set_automask_settings_for_caching(
        AutomaskSettings(enabled=True, alpha=128, shrink=0, blur=0),
        remover=remover,
    )
    try:
        _images, alpha_masks, _original_sizes, _crop_ltrbs = train_util.load_images_and_masks_for_caching(
            [info],
            use_alpha_mask=True,
            random_crop=False,
        )
    finally:
        train_util.set_automask_settings_for_caching(AutomaskSettings(enabled=False), remover=None)

    assert np.allclose(
        alpha_masks[0].numpy(),
        np.array([[1.0, 128 / 255.0], [128 / 255.0, 1.0]], dtype=np.float32),
    )
    assert np.array(remover.seen_image)[0, 1].tolist() == [255, 255, 255]


def test_latents_strategy_saves_automask_alpha_as_uint8_and_loads_float(tmp_path):
    npz_path = tmp_path / "latents.npz"
    strategy = LatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)
    alpha_mask = np.array([[0.0, 128 / 255.0, 1.0]], dtype=np.float32)

    strategy.save_latents_to_disk(
        str(npz_path),
        latents_tensor=np.zeros((4, 1, 1), dtype=np.float32),
        original_size=(3, 1),
        crop_ltrb=(0, 0, 0, 0),
        alpha_mask=alpha_mask,
        automask_settings=AutomaskSettings(enabled=True, alpha=128, shrink=0, blur=0),
    )

    with np.load(npz_path) as data:
        assert data["alpha_mask"].dtype == np.uint8
        assert data["alpha_mask"].tolist() == [[0, 128, 255]]

    _latents, _original_size, _crop_ltrb, _flipped, loaded_alpha = strategy._default_load_latents_from_disk(
        None,
        str(npz_path),
        (3, 1),
    )
    assert np.allclose(loaded_alpha, alpha_mask)


def test_latents_cache_validity_rejects_stale_automask_metadata(tmp_path):
    from library import train_util

    npz_path = tmp_path / "latents.npz"
    strategy = LatentsCachingStrategy(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False)
    strategy.save_latents_to_disk(
        str(npz_path),
        latents_tensor=np.zeros((4, 1, 1), dtype=np.float32),
        original_size=(1, 1),
        crop_ltrb=(0, 0, 0, 0),
        alpha_mask=np.ones((1, 1), dtype=np.float32),
        automask_settings=AutomaskSettings(enabled=True, alpha=128, shrink=1, blur=3, model="base-nightly"),
    )

    try:
        train_util.set_automask_settings_for_caching(
            AutomaskSettings(enabled=True, alpha=128, shrink=1, blur=3, model="base-nightly")
        )
        assert strategy._default_is_disk_cached_latents_expected(1, (1, 1), str(npz_path), False, True)

        train_util.set_automask_settings_for_caching(
            AutomaskSettings(enabled=True, alpha=64, shrink=1, blur=3, model="base-nightly")
        )
        assert not strategy._default_is_disk_cached_latents_expected(1, (1, 1), str(npz_path), False, True)
    finally:
        train_util.set_automask_settings_for_caching(AutomaskSettings(enabled=False))


def test_dataset_group_releases_automask_remover_after_new_cache_latents():
    from library import train_util

    class FakeDataset:
        image_data = {}
        num_train_images = 0
        num_reg_images = 0

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise IndexError(index)

        def new_cache_latents(self, model, accelerator):
            assert train_util.AUTOMASK_REMOVER is remover

    class FakeAccelerator:
        def __init__(self):
            self.waited = False

        def wait_for_everyone(self):
            self.waited = True

    remover = object()
    accelerator = FakeAccelerator()
    dataset_group = train_util.DatasetGroup([FakeDataset()])

    train_util.set_automask_settings_for_caching(AutomaskSettings(enabled=True), remover=remover)
    dataset_group.new_cache_latents(object(), accelerator)

    assert accelerator.waited is True
    assert train_util.AUTOMASK_REMOVER is None
    assert train_util.AUTOMASK_REMOVER_MODEL is None


def test_release_automask_remover_clears_cuda_cache(monkeypatch):
    from library import train_util

    calls = []
    remover = object()
    train_util.set_automask_settings_for_caching(AutomaskSettings(enabled=True), remover=remover)
    monkeypatch.setattr(train_util.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(train_util.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_util.torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    train_util.release_automask_remover()

    assert train_util.AUTOMASK_REMOVER is None
    assert train_util.AUTOMASK_REMOVER_MODEL is None
    assert calls == ["gc", "cuda"]


def test_automask_remover_uses_caption_device_selection(monkeypatch):
    from library import train_util

    created = []

    class FakeRemover:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(train_util.torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "transparent_background", type("FakeModule", (), {"Remover": FakeRemover}))
    train_util.release_automask_remover()

    remover = train_util._get_automask_remover(AutomaskSettings(enabled=True, model="base-nightly"))

    assert isinstance(remover, FakeRemover)
    assert created == [{"device": "cuda", "mode": "base-nightly"}]
    train_util.release_automask_remover()


def test_training_parser_accepts_automask_arguments():
    from library import train_util

    parser = argparse.ArgumentParser()
    train_util.add_training_arguments(parser, support_dreambooth=True)

    args = parser.parse_args(
        [
            "--automask",
            "--automask_alpha",
            "128",
            "--automask_shrink",
            "2",
            "--automask_blur",
            "4",
            "--automask_model",
            "base-nightly",
        ]
    )

    assert args.automask is True
    assert args.automask_alpha == 128
    assert args.automask_shrink == 2
    assert args.automask_blur == 4
    assert args.automask_model == "base-nightly"


@pytest.mark.parametrize("timestep_sampling", ["autoshift", "autoshift_wavelet"])
def test_autoshift_enables_mask_generation_without_enabling_automask_loss(timestep_sampling):
    from library import train_util

    args = argparse.Namespace(
        automask=False,
        timestep_sampling=timestep_sampling,
        automask_alpha=128,
        automask_shrink=1,
        automask_blur=3,
        automask_model="base-nightly",
    )
    train_util.configure_automask_from_args(args)

    assert train_util.get_automask_settings_for_caching().enabled is True
    assert args.automask is False
    train_util.set_automask_settings_for_caching(AutomaskSettings(enabled=False))
