import argparse
import unittest

import torch

from library import anima_train_utils


class DifferentialGuidanceTest(unittest.TestCase):
    def test_anima_parser_accepts_differential_guidance_scale(self):
        parser = argparse.ArgumentParser()
        anima_train_utils.add_anima_training_arguments(parser)

        args = parser.parse_args(["--differential_guidance_scale", "1.5"])

        self.assertEqual(args.differential_guidance_scale, 1.5)

    def test_differential_guidance_scale_one_keeps_target(self):
        model_pred = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[3.0, 5.0]]]])

        guided = anima_train_utils.apply_differential_guidance_target(target, model_pred, 1.0)

        self.assertIs(guided, target)
        self.assertTrue(torch.equal(guided, target))

    def test_differential_guidance_extrapolates_from_detached_prediction(self):
        model_pred = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[3.0, 5.0]]]], requires_grad=True)

        guided = anima_train_utils.apply_differential_guidance_target(target, model_pred, 1.5)

        expected = model_pred.detach() + 1.5 * (target - model_pred.detach())
        self.assertTrue(torch.equal(guided, expected))

        guided.sum().backward()
        self.assertIsNone(model_pred.grad)
        self.assertIsNotNone(target.grad)

    def test_logit_normal_weighting_is_non_uniform_and_normalized(self):
        args = argparse.Namespace(logit_mean=0.0, logit_std=1.0)
        sigmas = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        weighting = anima_train_utils.compute_loss_weighting_for_anima("logit_normal", sigmas, args)

        self.assertFalse(torch.allclose(weighting, torch.ones_like(weighting)))
        self.assertAlmostEqual(weighting.mean().item(), 1.0, places=5)
        self.assertGreater(weighting[2].item(), weighting[0].item())

    def test_mode_weighting_uses_mode_scale(self):
        sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        low = anima_train_utils.compute_loss_weighting_for_anima(
            "mode", sigmas, argparse.Namespace(mode_scale=0.1)
        )
        high = anima_train_utils.compute_loss_weighting_for_anima(
            "mode", sigmas, argparse.Namespace(mode_scale=2.0)
        )

        self.assertGreater(high[2].item() - high[0].item(), low[2].item() - low[0].item())
        self.assertAlmostEqual(high.mean().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
