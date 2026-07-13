import pytest
import torch
from unittest.mock import MagicMock, patch
from library.flux_train_utils import get_noisy_model_input_and_timesteps as flux_get_noisy
from library.lumina_train_util import get_noisy_model_input_and_timesteps as lumina_get_noisy
from library.sd3_train_utils import get_noisy_model_input_and_timesteps as sd3_get_noisy

class MockNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = MagicMock()
        self.config.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps, dtype=torch.long)

@pytest.fixture
def args():
    args = MagicMock()
    args.timestep_sampling = "shift"
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 1.0
    args.training_shift = 1.0
    args.min_timestep = 0
    args.max_timestep = 1000
    args.weighting_scheme = "uniform"
    args.logit_mean = 0.0
    args.logit_std = 1.0
    args.mode_scale = 1.0
    return args

def test_flux_folder_shift(args):
    noise_scheduler = MockNoiseScheduler()
    latents = torch.randn(4, 4, 8, 8)
    noise = torch.randn(4, 4, 8, 8)
    device = "cpu"
    dtype = torch.float32

    with patch("library.flux_train_utils.torch.randn", return_value=torch.zeros(4)):
        _, timesteps, sigmas = flux_get_noisy(
            args, noise_scheduler, latents, noise, device, dtype,
            folder_shifts=["high", "low", "mid", "global"]
        )
        
        sigmas_flat = sigmas.flatten().tolist()
        
        assert pytest.approx(sigmas_flat[0], abs=1e-4) == 0.6
        assert pytest.approx(sigmas_flat[1], abs=1e-4) == 1.0 / 3.0
        assert pytest.approx(sigmas_flat[2], abs=1e-4) == 0.5
        assert pytest.approx(sigmas_flat[3], abs=1e-4) == 0.5

def test_lumina_folder_shift(args):
    noise_scheduler = MockNoiseScheduler()
    latents = torch.randn(4, 4, 8, 8)
    noise = torch.randn(4, 4, 8, 8)
    device = "cpu"
    dtype = torch.float32

    with patch("library.lumina_train_util.torch.randn", return_value=torch.zeros(4)):
        _, timesteps, sigmas = lumina_get_noisy(
            args, noise_scheduler, latents, noise, device, dtype,
            folder_shifts=["high", "low", "mid", "global"]
        )
        
        sigmas_flat = sigmas.flatten().tolist()
        
        assert pytest.approx(sigmas_flat[0], abs=1e-4) == 0.6
        assert pytest.approx(sigmas_flat[1], abs=1e-4) == 1.0 / 3.0
        assert pytest.approx(sigmas_flat[2], abs=1e-4) == 0.5
        assert pytest.approx(sigmas_flat[3], abs=1e-4) == 0.5

def test_sd3_folder_shift(args):
    latents = torch.randn(4, 4, 8, 8)
    noise = torch.randn(4, 4, 8, 8)
    device = "cpu"
    dtype = torch.float32

    with patch("library.sd3_train_utils.compute_density_for_timestep_sampling", return_value=torch.tensor([0.5, 0.5, 0.5, 0.5])):
        _, timesteps, sigmas = sd3_get_noisy(
            args, latents, noise, device, dtype,
            folder_shifts=["high", "low", "mid", "global"]
        )
        
        sigmas_flat = sigmas.flatten().tolist()
        
        assert pytest.approx(sigmas_flat[0], abs=1e-2) == 0.6
        assert pytest.approx(sigmas_flat[1], abs=1e-2) == 1.0 / 3.0
        assert pytest.approx(sigmas_flat[2], abs=1e-2) == 0.5
        assert pytest.approx(sigmas_flat[3], abs=1e-2) == 0.5
