import argparse
import unittest
from types import SimpleNamespace

import torch

from library import train_util


class FlowMatchDummyScheduler:
    pass


def make_args(**overrides):
    values = {
        "wavelet_loss_c_min": 1.0,
        "wavelet_loss_c_max": 1.0,
        "wavelet_loss_alpha": 0.5,
        "wavelet_loss_beta": 1.0,
        "wavelet_loss_gamma": 999.0,
        "wavelet_loss_weight": 1.0,
        "wavelet_loss_prediction_type": "sample",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SnrAwareHuberWaveletLossTest(unittest.TestCase):
    def test_training_parser_accepts_snr_aware_huber_wavelet_options(self):
        parser = argparse.ArgumentParser()
        train_util.add_training_arguments(parser, False)
        args = parser.parse_args(
            [
                "--loss_type",
                "snr_aware_huber_wavelet",
                "--wavelet_loss_beta",
                "0.25",
                "--wavelet_loss_prediction_type",
                "sample",
            ]
        )

        self.assertEqual(args.loss_type, "snr_aware_huber_wavelet")
        self.assertEqual(args.wavelet_loss_beta, 0.25)
        self.assertEqual(args.wavelet_loss_prediction_type, "sample")

    def test_training_parser_accepts_wavelet_l2(self):
        parser = argparse.ArgumentParser()
        train_util.add_training_arguments(parser, False)
        args = parser.parse_args(["--loss_type", "wavelet_l2"])

        self.assertEqual(args.loss_type, "wavelet_l2")

    def test_training_parser_accepts_ait_wavelet(self):
        parser = argparse.ArgumentParser()
        train_util.add_training_arguments(parser, False)
        args = parser.parse_args(["--loss_type", "wavelet"])

        self.assertEqual(args.loss_type, "wavelet")

    def test_wavelet_matches_ait_noise_minus_prediction_residual(self):
        latents = torch.zeros(1, 1, 4, 4)
        noise = torch.ones_like(latents)
        model_pred = torch.zeros_like(latents)
        model_pred[:, :, 0, 0] = 0.25

        loss = train_util.conditional_loss(
            model_pred,
            noise,
            "wavelet",
            "none",
            latents=latents,
            noise=noise,
        )

        expected = (train_util.haar_dwt_2d(noise - model_pred) - train_util.haar_dwt_2d(latents)).pow(2)
        self.assertTrue(torch.equal(loss, expected))

    def test_wavelet_l2_matches_haar_squared_residual(self):
        latents = torch.zeros(1, 1, 4, 4)
        model_pred = torch.zeros_like(latents)
        model_pred[:, :, 0, 0] = 1.0
        sigmas = torch.full((1, 1, 1, 1), 0.5)
        noisy_latents = model_pred
        timesteps = sigmas.flatten() * 1000

        loss = train_util.conditional_loss(
            model_pred,
            model_pred,
            "wavelet_l2",
            "none",
            latents=latents,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            sigmas=sigmas,
            noise_scheduler=FlowMatchDummyScheduler(),
            args=make_args(wavelet_loss_prediction_type="sample"),
        )

        expected = train_util.haar_dwt_2d(model_pred - latents).pow(2)
        self.assertTrue(torch.equal(loss, expected))

    def test_snr_aware_huber_wavelet_applies_ultraflux_time_weight(self):
        latents = torch.zeros(2, 1, 2, 2)
        z_pred = torch.ones_like(latents)
        noisy_latents = torch.zeros_like(latents)
        sigmas = torch.tensor([0.25, 0.5]).view(2, 1, 1, 1)
        timesteps = sigmas.flatten() * 1000

        loss = train_util.snr_aware_huber_wavelet_loss(
            z_pred,
            z_pred,
            latents,
            noisy_latents,
            timesteps,
            sigmas,
            FlowMatchDummyScheduler(),
            make_args(),
        )

        per_sample = loss.mean(dim=(1, 2, 3))
        self.assertTrue(torch.isclose(per_sample[0] / per_sample[1], torch.tensor(3.0), atol=1e-5))

    def test_snr_aware_huber_wavelet_squeezes_singleton_frame_noisy_latents(self):
        batch_size = 2
        latents = torch.zeros(batch_size, 1, 4, 4)
        noise = torch.ones_like(latents)
        sigmas = torch.full((batch_size, 1, 1, 1), 0.5)
        noisy_latents = ((1.0 - sigmas) * latents + sigmas * noise).unsqueeze(2)
        timesteps = sigmas.flatten() * 1000

        loss = train_util.conditional_loss(
            noise,
            noise,
            "snr_aware_huber_wavelet",
            "none",
            latents=latents,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            sigmas=sigmas,
            noise_scheduler=FlowMatchDummyScheduler(),
            args=make_args(wavelet_loss_prediction_type="velocity"),
        )

        self.assertEqual(loss.shape, (batch_size, 4, 2, 2))
        self.assertTrue(torch.isfinite(loss).all())

    def test_snr_aware_huber_wavelet_prediction_types_reconstruct_clean_latents(self):
        cases = [
            ("velocity", torch.ones(2, 1, 4, 4)),
            ("negative_velocity", -torch.ones(2, 1, 4, 4)),
            ("sample", torch.zeros(2, 1, 4, 4)),
        ]
        for prediction_type, model_pred in cases:
            with self.subTest(prediction_type=prediction_type):
                batch_size = 2
                latents = torch.zeros(batch_size, 1, 4, 4)
                noise = torch.ones_like(latents)
                sigmas = torch.full((batch_size, 1, 1, 1), 0.5)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                timesteps = sigmas.flatten() * 1000

                loss = train_util.conditional_loss(
                    model_pred,
                    model_pred,
                    "snr_aware_huber_wavelet",
                    "none",
                    latents=latents,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    sigmas=sigmas,
                    noise_scheduler=FlowMatchDummyScheduler(),
                    args=make_args(wavelet_loss_prediction_type=prediction_type),
                )

                self.assertEqual(torch.count_nonzero(loss), 0)


if __name__ == "__main__":
    unittest.main()
