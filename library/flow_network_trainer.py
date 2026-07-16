"""Shared rectified-flow network training behavior for Anima-family models."""

import argparse
import logging
import math
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import torch

from library import anima_train_utils, cdc_fm, qwen_image_autoencoder_kl, sd3_train_utils, self_flow, train_util


logger = logging.getLogger(__name__)


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
    parser.add_argument("--use_cdc_fm", action="store_true", help="enable Carré du champ flow matching")
    parser.add_argument("--cdc_k_neighbors", type=int, default=64)
    parser.add_argument("--cdc_k_bandwidth", type=int, default=8)
    parser.add_argument("--cdc_dim", type=int, default=8)
    parser.add_argument("--cdc_gamma", type=float, default=1.0)
    parser.add_argument("--cdc_bandwidth_rescale", type=float, default=1.0)
    parser.add_argument("--cdc_min_bucket_size", type=int, default=8)
    parser.add_argument("--cdc_cache_dir", type=str, default="tasks/cdc_cache")
    parser.add_argument("--cdc_cache_memory_entries", type=int, default=32)
    parser.add_argument("--cdc_force_recache", action="store_true")
    parser.add_argument(
        "--cdc_combine_knn",
        action="store_true",
        help="apply CDC geometry to the KNN-selected Immiscible Diffusion noise in the same training step",
    )
    parser.add_argument("--cdc_alternate_knn", action="store_true", help="alternate KNN-only and CDC-only optimizer steps")
    parser.add_argument("--cdc_knn_metric", choices=["l2", "mahalanobis"], default="l2")
    parser.add_argument("--cdc_knn_regularization", type=float, default=0.1)
    parser.add_argument(
        "--cdc_switch_ratio",
        type=float,
        default=0.0,
        help="fraction of training steps using KNN noise before switching to CDC-FM (0 disables scheduling)",
    )
    parser.add_argument("--use_self_flow", action="store_true", help="enable Self-Flow dual-timestep self-distillation")
    parser.add_argument("--self_flow_mask_ratio", type=float, default=0.25)
    parser.add_argument("--self_flow_representation_weight", type=float, default=0.8)
    parser.add_argument("--self_flow_ema_decay", type=float, default=0.9999)
    parser.add_argument("--self_flow_student_layer", type=int, default=8)
    parser.add_argument("--self_flow_teacher_layer", type=int, default=20)
    parser.add_argument("--self_flow_projection_dim", type=int, default=768)
    parser.add_argument("--self_flow_projection_lr", type=float, default=None)
    parser.add_argument(
        "--self_flow_save_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save EMA adapter parameters in inference checkpoints",
    )


class FlowNetworkTrainerMixin:
    """Common scheduler, target shaping, validation, and metadata behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = 0
        self.cdc_geometry_cache = None
        self._cdc_last_stage = None
        self.self_flow_ema = None
        self.current_self_flow_representation_loss = None

    def step_logging(self, accelerator, logs, global_step, epoch):
        self.global_step = global_step
        if self.current_self_flow_representation_loss is not None:
            logs["self_flow/representation_loss"] = float(
                self.current_self_flow_representation_loss.detach().float().item()
            )
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
        if getattr(args, "use_self_flow", False):
            if not args.network_train_unet_only or args.network_train_text_encoder_only:
                raise ValueError("Self-Flow network training requires --network_train_unet_only")
            if getattr(args, "use_cdc_fm", False):
                raise ValueError("Self-Flow and CDC-FM alter the flow path and cannot be enabled together")
            if not 0.0 < args.self_flow_mask_ratio <= 0.5:
                raise ValueError("self_flow_mask_ratio must be in (0, 0.5]")
            if args.self_flow_representation_weight < 0.0:
                raise ValueError("self_flow_representation_weight must be greater than or equal to 0")
            if not 0.0 <= args.self_flow_ema_decay < 1.0:
                raise ValueError("self_flow_ema_decay must be in [0, 1)")
            if args.self_flow_student_layer < 0 or args.self_flow_teacher_layer < 0:
                raise ValueError("Self-Flow layer indices must be greater than or equal to 0")
            if args.self_flow_student_layer >= args.self_flow_teacher_layer:
                raise ValueError("Self-Flow student layer must be shallower than teacher layer")
            if args.self_flow_projection_dim < 1:
                raise ValueError("self_flow_projection_dim must be greater than 0")
            if args.self_flow_projection_lr is not None and args.self_flow_projection_lr <= 0.0:
                raise ValueError("self_flow_projection_lr must be greater than 0")
            if args.network_dropout not in (None, 0.0):
                raise ValueError("Self-Flow does not support network_dropout because the EMA teacher must be deterministic")
            if getattr(args, "ip_noise_gamma", 0.0):
                raise ValueError("Self-Flow does not support ip_noise_gamma because it changes the dual-timestep path")
            for network_arg in args.network_args or []:
                if network_arg.startswith(("rank_dropout=", "module_dropout=")):
                    value = float(network_arg.split("=", 1)[1])
                    if value != 0.0:
                        raise ValueError("Self-Flow does not support rank_dropout or module_dropout")
        if getattr(args, "use_cdc_fm", False):
            switch_ratio = float(getattr(args, "cdc_switch_ratio", 0.0))
            combine_knn = bool(getattr(args, "cdc_combine_knn", False))
            alternate_knn = bool(getattr(args, "cdc_alternate_knn", False))
            if not 0.0 <= switch_ratio < 1.0:
                raise ValueError("cdc_switch_ratio must be in [0, 1)")
            if sum((combine_knn, alternate_knn, switch_ratio > 0.0)) > 1:
                raise ValueError("CDC combined, alternate, and switch modes are mutually exclusive")
            if (combine_knn or alternate_knn or switch_ratio > 0.0) and getattr(args, "knn_noise_k", 0) < 1:
                raise ValueError("CDC combined, alternate, or switch mode requires knn_noise_k to be greater than 0")
            if args.cdc_knn_regularization <= 0.0:
                raise ValueError("cdc_knn_regularization must be greater than 0")
            if not args.cache_latents or not args.cache_latents_to_disk:
                raise ValueError("CDC-FM requires --cache_latents and --cache_latents_to_disk")
            if args.cdc_k_neighbors < 2:
                raise ValueError("cdc_k_neighbors must be greater than or equal to 2")
            if args.cdc_k_bandwidth < 1:
                raise ValueError("cdc_k_bandwidth must be greater than or equal to 1")
            if args.cdc_dim < 1:
                raise ValueError("cdc_dim must be greater than or equal to 1")
            if args.cdc_gamma <= 0.0:
                raise ValueError("cdc_gamma must be greater than 0")
            if args.cdc_bandwidth_rescale <= 0.0:
                raise ValueError("cdc_bandwidth_rescale must be greater than 0")
            if args.cdc_min_bucket_size < 3:
                raise ValueError("cdc_min_bucket_size must be greater than or equal to 3")
            if args.cdc_cache_memory_entries < 1:
                raise ValueError("cdc_cache_memory_entries must be greater than or equal to 1")

    def prepare_after_latents_cached(self, args, train_dataset_group, accelerator) -> None:
        if not getattr(args, "use_cdc_fm", False):
            return
        architecture = getattr(self, "cdc_architecture_name", self.__class__.__name__.lower())
        self.cdc_geometry_cache = cdc_fm.prepare_cdc_cache(args, train_dataset_group, accelerator, architecture)

    def set_current_training_step(self, global_step: int) -> None:
        self.global_step = int(global_step)
        super().set_current_training_step(global_step)

    def is_cdc_stage_active(self, args, is_train: bool = True) -> bool:
        if not is_train or not getattr(args, "use_cdc_fm", False):
            return False
        if getattr(args, "cdc_alternate_knn", False):
            return self.global_step % 2 == 1
        switch_ratio = float(getattr(args, "cdc_switch_ratio", 0.0))
        if switch_ratio <= 0.0:
            return True
        switch_step = math.ceil(int(args.max_train_steps) * switch_ratio)
        return self.global_step >= switch_step

    def sample_flow_training_noise(self, args, latents: torch.Tensor, batch=None, is_train: bool = True) -> torch.Tensor:
        if not getattr(args, "use_cdc_fm", False):
            return train_util.sample_training_noise(args, latents)

        cdc_active = self.is_cdc_stage_active(args, is_train)
        switch_ratio = float(getattr(args, "cdc_switch_ratio", 0.0))
        combine_knn = bool(getattr(args, "cdc_combine_knn", False))
        alternate_knn = bool(getattr(args, "cdc_alternate_knn", False))
        if is_train and cdc_active and combine_knn:
            stage = "combined"
        elif cdc_active:
            stage = "cdc"
        elif is_train and (switch_ratio > 0.0 or alternate_knn):
            stage = "knn"
        else:
            stage = "gaussian"
        if is_train and alternate_knn:
            if self._cdc_last_stage != "alternate":
                logger.info("CDC-FM: alternating KNN-only and CDC-only optimizer steps")
            self._cdc_last_stage = "alternate"
        elif is_train and stage != self._cdc_last_stage:
            if stage == "knn":
                switch_step = math.ceil(int(args.max_train_steps) * switch_ratio)
                logger.info("CDC-FM schedule: using KNN noise until step %d", switch_step)
            elif stage == "cdc":
                logger.info("CDC-FM schedule: switched to CDC geometry at step %d", self.global_step)
            elif stage == "combined":
                logger.info("CDC-FM: combining KNN Immiscible noise with CDC geometry")
            self._cdc_last_stage = stage

        if stage in ("knn", "combined"):
            if getattr(args, "cdc_knn_metric", "l2") == "mahalanobis":
                if self.cdc_geometry_cache is None:
                    raise RuntimeError("CDC-aware KNN requires a prepared CDC geometry cache")
                if batch is None or batch.get("cdc_keys") is None:
                    raise RuntimeError("CDC-aware KNN training batch does not contain cdc_keys")
                candidates = torch.randn(
                    (latents.shape[0], int(args.knn_noise_k), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                return self.cdc_geometry_cache.select_geometry_aware_noise(
                    latents,
                    candidates,
                    batch["cdc_keys"],
                    batch.get("flippeds"),
                    float(args.cdc_knn_regularization),
                )
            return train_util.sample_knn_noise(latents, int(args.knn_noise_k))
        return torch.randn_like(latents, device=latents.device)

    def apply_cdc_flow_path(self, args, batch, latents, noise, sigmas, is_train):
        if not self.is_cdc_stage_active(args, is_train):
            return None
        if self.cdc_geometry_cache is None:
            raise RuntimeError("CDC-FM is enabled but its geometry cache was not prepared")
        keys = batch.get("cdc_keys")
        if keys is None:
            raise RuntimeError("CDC-FM training batch does not contain cdc_keys")
        correction = self.cdc_geometry_cache.correction(noise, keys, batch.get("flippeds"))
        return cdc_fm.apply_cdc_flow_path(latents, noise, sigmas, correction)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device):
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return train_util.apply_immiscible_image_scale(args, latents)

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if getattr(args, "use_self_flow", False) and self.current_self_flow_representation_loss is not None:
            loss = loss + args.self_flow_representation_weight * self.current_self_flow_representation_loss
        return loss

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        if not getattr(args, "use_self_flow", False):
            return
        feature_dim = getattr(unet, "model_channels", None)
        if feature_dim is None and hasattr(unet, "config"):
            feature_dim = getattr(unet.config, "features", None)
        if feature_dim is None:
            raise ValueError("Self-Flow could not determine the DiT feature dimension")
        block_count = len(getattr(unet, "blocks", ()))
        if args.self_flow_teacher_layer >= block_count:
            raise ValueError(
                f"self_flow_teacher_layer={args.self_flow_teacher_layer} exceeds the {block_count}-block model"
            )
        projector = self_flow.ProjectionHead(feature_dim, args.self_flow_projection_dim)
        projector.to(device=accelerator.device)
        network.add_module(self_flow.PROJECTOR_MODULE_NAME, projector)

    def on_network_weights_loaded(self, args, accelerator, network, text_encoders, unet):
        super().on_network_weights_loaded(args, accelerator, network, text_encoders, unet)
        if not getattr(args, "use_self_flow", False):
            return
        named_parameters = self_flow.dit_adapter_named_parameters(network)
        self.self_flow_ema = self_flow.AdapterEMA(named_parameters, args.self_flow_ema_decay)
        accelerator.register_for_checkpointing(self.self_flow_ema)

    def get_additional_optimizer_params(self, args, network):
        groups, descriptions = super().get_additional_optimizer_params(args, network)
        if not getattr(args, "use_self_flow", False):
            return groups, descriptions
        projector = getattr(network, self_flow.PROJECTOR_MODULE_NAME)
        lr = args.self_flow_projection_lr
        if lr is None:
            lr = args.unet_lr if args.unet_lr is not None else args.learning_rate
        return list(groups) + [{"params": projector.parameters(), "lr": lr}], list(descriptions) + ["self-flow projector"]

    def on_optimizer_step(self, args, accelerator, network):
        super().on_optimizer_step(args, accelerator, network)
        if getattr(args, "use_self_flow", False):
            self.self_flow_ema.update()

    def network_save_context(self, args, network):
        if not getattr(args, "use_self_flow", False):
            return super().network_save_context(args, network)
        return self_flow.adapter_output_context(network, self.self_flow_ema, args.self_flow_save_ema)

    @contextmanager
    def self_flow_teacher_context(self, network, model=None):
        """Use deterministic EMA adapter weights for the no-grad teacher pass."""

        if self.self_flow_ema is None:
            raise RuntimeError("Self-Flow EMA was not initialized")
        was_training = network.training
        model_was_training = model.training if model is not None else None
        network.eval()
        if model is not None:
            model.eval()
        try:
            with self.self_flow_ema.apply():
                yield
        finally:
            if model is not None:
                model.train(model_was_training)
            network.train(was_training)

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
        proposal_flow_shift: Optional[Union[float, torch.Tensor]] = None,
        base_target: Optional[torch.Tensor] = None,
    ):
        target = self.apply_model_guidance_target(
            args, noise - latents if base_target is None else base_target, model_pred, model_pred_uncond, timestep_fractions
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
            proposal_flow_shift=proposal_flow_shift,
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
        metadata["ss_use_cdc_fm"] = getattr(args, "use_cdc_fm", False)
        metadata["ss_cdc_k_neighbors"] = getattr(args, "cdc_k_neighbors", 0)
        metadata["ss_cdc_k_bandwidth"] = getattr(args, "cdc_k_bandwidth", 0)
        metadata["ss_cdc_dim"] = getattr(args, "cdc_dim", 0)
        metadata["ss_cdc_gamma"] = getattr(args, "cdc_gamma", 0.0)
        metadata["ss_cdc_bandwidth_rescale"] = getattr(args, "cdc_bandwidth_rescale", 0.0)
        metadata["ss_cdc_min_bucket_size"] = getattr(args, "cdc_min_bucket_size", 0)
        metadata["ss_cdc_combine_knn"] = getattr(args, "cdc_combine_knn", False)
        metadata["ss_cdc_alternate_knn"] = getattr(args, "cdc_alternate_knn", False)
        metadata["ss_cdc_knn_metric"] = getattr(args, "cdc_knn_metric", "l2")
        metadata["ss_cdc_knn_regularization"] = getattr(args, "cdc_knn_regularization", 0.1)
        metadata["ss_cdc_switch_ratio"] = getattr(args, "cdc_switch_ratio", 0.0)
        metadata["ss_use_self_flow"] = getattr(args, "use_self_flow", False)
        metadata["ss_self_flow_mask_ratio"] = getattr(args, "self_flow_mask_ratio", 0.0)
        metadata["ss_self_flow_representation_weight"] = getattr(args, "self_flow_representation_weight", 0.0)
        metadata["ss_self_flow_ema_decay"] = getattr(args, "self_flow_ema_decay", 0.0)
        metadata["ss_self_flow_student_layer"] = getattr(args, "self_flow_student_layer", 0)
        metadata["ss_self_flow_teacher_layer"] = getattr(args, "self_flow_teacher_layer", 0)
        metadata["ss_self_flow_projection_dim"] = getattr(args, "self_flow_projection_dim", 0)
        metadata["ss_self_flow_save_ema"] = getattr(args, "self_flow_save_ema", False)
