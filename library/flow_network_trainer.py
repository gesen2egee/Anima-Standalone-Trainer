"""Shared rectified-flow network training behavior for Anima-family models."""

import argparse
import math
from typing import Optional, Sequence

import torch

from library import anima_train_utils, qwen_image_autoencoder_kl, sd3_train_utils, train_util


def add_flow_network_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Register options shared by Anima and Krea 2 network training."""
    parser.add_argument("--model_guidance_weight", type=float, default=0.0)
    parser.add_argument("--model_guidance_warmup_steps", type=int, default=500)
    parser.add_argument("--model_guidance_timestep_scaling", action="store_true")
    parser.add_argument("--model_guidance_min_weight", type=float, default=0.0)
    parser.add_argument("--model_guidance_cfg_zero", action="store_true")
    parser.add_argument("--model_guidance_zero_init_threshold", type=float, default=1.0)
    parser.add_argument("--model_guidance_end_step", type=int, default=0)
    parser.add_argument("--ciop_noise_magnitude", type=float, default=0.1)
    parser.add_argument("--ciop_noise_type", choices=["gaussian", "uniform"], default="gaussian")


class FlowNetworkTrainerMixin:
    """Common scheduler, target shaping, validation, and metadata behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = 0

    def step_logging(self, accelerator, logs, global_step, epoch):
        self.global_step = global_step
        super().step_logging(accelerator, logs, global_step, epoch)

    @staticmethod
    def validate_flow_training_args(args) -> None:
        if not 0.0 <= args.model_guidance_prob <= 1.0:
            raise ValueError("model_guidance_prob must be between 0.0 and 1.0")
        if not 0.0 <= args.ciop_prob <= 1.0:
            raise ValueError("ciop_prob must be between 0.0 and 1.0")
        if args.ciop_noise_magnitude < 0.0:
            raise ValueError("ciop_noise_magnitude must be greater than or equal to 0.0")
        if args.model_guidance_weight < 0.0:
            raise ValueError("model_guidance_weight must be greater than or equal to 0.0")
        if args.model_guidance_warmup_steps < 0:
            raise ValueError("model_guidance_warmup_steps must be greater than or equal to 0")
        if args.model_guidance_end_step < 0:
            raise ValueError("model_guidance_end_step must be greater than or equal to 0")
        if not 0.0 <= args.model_guidance_zero_init_threshold <= 1.0:
            raise ValueError("model_guidance_zero_init_threshold must be between 0.0 and 1.0")

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device):
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return train_util.apply_immiscible_image_scale(args, latents)

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    @staticmethod
    def sample_ciop_perturbations(args, reference: torch.Tensor, is_train: bool):
        if not is_train or args.ciop_prob <= 0.0:
            return None
        if torch.rand((), device=reference.device).item() >= args.ciop_prob:
            return None
        if args.ciop_noise_type == "uniform":
            limit = math.sqrt(3.0) * args.ciop_noise_magnitude

            def sample():
                return (torch.rand_like(reference) * 2.0 - 1.0) * limit
        else:
            def sample():
                return torch.randn_like(reference) * args.ciop_noise_magnitude
        return sample(), sample()

    def should_apply_model_guidance(self, args, device: torch.device) -> bool:
        if args.model_guidance_weight <= 0.0:
            return False
        if args.model_guidance_end_step > 0 and self.global_step >= args.model_guidance_end_step:
            return False
        return args.model_guidance_prob >= 1.0 or torch.rand((), device=device).item() <= args.model_guidance_prob

    def apply_model_guidance_target(
        self,
        args,
        target: torch.Tensor,
        model_pred: torch.Tensor,
        model_pred_uncond: Optional[torch.Tensor],
        timestep_fractions: torch.Tensor,
    ) -> torch.Tensor:
        if model_pred_uncond is None:
            return target
        current_weight = args.model_guidance_weight
        if args.model_guidance_warmup_steps > 0 and self.global_step < args.model_guidance_warmup_steps:
            current_weight *= self.global_step / args.model_guidance_warmup_steps

        shape = (-1, *([1] * (model_pred.ndim - 1)))
        if args.model_guidance_cfg_zero:
            reduce_dims = tuple(range(1, model_pred.ndim))
            dot = torch.sum(model_pred * model_pred_uncond, dim=reduce_dims, keepdim=True)
            norm = torch.sum(model_pred_uncond.square(), dim=reduce_dims, keepdim=True) + 1e-8
            guidance = current_weight * (dot / norm).detach()
            active = timestep_fractions < args.model_guidance_zero_init_threshold
            guidance = guidance * active.to(model_pred.dtype).view(shape)
        elif args.model_guidance_timestep_scaling:
            beta_curve = 4.0 * timestep_fractions * (1.0 - timestep_fractions)
            guidance = args.model_guidance_min_weight + (current_weight - args.model_guidance_min_weight) * beta_curve
            guidance = guidance.view(shape)
        else:
            guidance = current_weight
        return target + guidance * (model_pred - model_pred_uncond).detach()

    def finalize_flow_target(
        self,
        args,
        noise: torch.Tensor,
        latents: torch.Tensor,
        model_pred: torch.Tensor,
        sigmas: torch.Tensor,
        timestep_fractions: torch.Tensor,
        model_pred_uncond: Optional[torch.Tensor] = None,
        ciop_output: Optional[torch.Tensor] = None,
        folder_shifts: Optional[Sequence[str]] = None,
        folder_shift_progress: Optional[float] = None,
    ):
        target = self.apply_model_guidance_target(
            args, noise - latents, model_pred, model_pred_uncond, timestep_fractions
        )
        if ciop_output is not None:
            target = target + ciop_output
        target = anima_train_utils.apply_differential_guidance_target(
            target, model_pred, args.differential_guidance_scale
        )
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme,
            sigmas=sigmas,
            args=args,
            folder_shifts=folder_shifts,
            folder_shift_progress=folder_shift_progress,
        )
        return target, weighting

    @staticmethod
    def update_flow_metadata(metadata, args) -> None:
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_model_guidance_prob"] = args.model_guidance_prob
        metadata["ss_model_guidance_weight"] = args.model_guidance_weight
        metadata["ss_differential_guidance_scale"] = args.differential_guidance_scale
        metadata["ss_ciop_prob"] = args.ciop_prob
        metadata["ss_ciop_noise_magnitude"] = args.ciop_noise_magnitude
        metadata["ss_ciop_noise_type"] = args.ciop_noise_type
        metadata["ss_knn_noise_k"] = getattr(args, "knn_noise_k", 0)
        metadata["ss_immiscible_image_scale"] = getattr(args, "immiscible_image_scale", 1.0)
