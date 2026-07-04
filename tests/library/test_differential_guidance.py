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


if __name__ == "__main__":
    unittest.main()
