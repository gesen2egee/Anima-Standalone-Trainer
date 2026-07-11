"""Krea 2 LoRA training through the existing Anima/sd-scripts pipeline."""

import argparse
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator

import train_network
from library import anima_train_utils, flux_train_utils, qwen_image_autoencoder_kl, sd3_train_utils, train_util
from library.device_utils import clean_memory_on_device
from library.krea2 import krea2_sampling, krea2_utils
from library.strategy_krea2 import (
    Krea2LatentsCachingStrategy,
    Krea2TextEncoderOutputsCachingStrategy,
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
)

logger = logging.getLogger(__name__)


class Krea2NetworkTrainer(train_network.NetworkTrainer):
    """Architecture adapter that keeps the shared optimizer/dataset/checkpoint loop."""

    def __init__(self):
        super().__init__()
        self.is_swapping_blocks = False
        self.global_step = 0

    def step_logging(self, accelerator, logs, global_step, epoch):
        self.global_step = global_step
        super().step_logging(accelerator, logs, global_step, epoch)

    @staticmethod
    def _dataset_needs_dynamic_caption_encoding(dataset) -> bool:
        """Return whether caption processing changes between training samples/steps."""
        for subset in getattr(dataset, "subsets", []):
            if (
                subset.caption_dropout_rate > 0
                or subset.caption_dropout_every_n_epochs > 0
                or subset.shuffle_caption
                or subset.token_warmup_step > 0
                or subset.caption_tag_dropout_rate > 0
                or getattr(subset, "enable_fad", False)
                or subset.caption_prefix
                or subset.caption_suffix
                or subset.enable_wildcard
            ):
                return True
        return False

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
        dynamic_caption_encoding = self._dataset_needs_dynamic_caption_encoding(train_dataset_group)
        if val_dataset_group is not None:
            dynamic_caption_encoding = dynamic_caption_encoding or self._dataset_needs_dynamic_caption_encoding(val_dataset_group)

        dynamic_text_encoder = (
            getattr(args, "krea2_dynamic_text_encoder", False)
            or getattr(args, "krea2_dynamic_text_encoder_cpu", False)
        )

        if dynamic_caption_encoding or dynamic_text_encoder:
            logger.warning(
                "Krea 2 dynamic text encoding is enabled; disabling text output cache and encoding captions during training."
            )
            args.cache_text_encoder_outputs = False
            args.cache_text_encoder_outputs_to_disk = False
        elif not args.cache_text_encoder_outputs:
            raise ValueError(
                "Krea 2 requires --cache_text_encoder_outputs. "
                "Run the text-cache stage before starting training."
            )
        if not args.cache_latents:
            raise ValueError(
                "Krea 2 requires --cache_latents because the shared trainer keeps the Qwen-Image VAE on CPU."
            )
        if args.network_train_text_encoder_only:
            raise ValueError("Krea 2 text encoder is frozen; --network_train_text_encoder_only is not supported")
        if args.train_inpainting:
            raise ValueError("Krea 2 supports masked loss, but the Anima inpainting input path is not compatible")

        args.network_train_unet_only = True
        # Krea 2 only supports the safe scaled-FP8 path. Keep the shared
        # fp8_base flag as the compatibility gate, but reject plain FP8 rather
        # than silently running in a different precision than the UI requests.
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Krea 2 requires FP8 Scaled together with FP8 Base; plain FP8 is not supported.")
        if args.fp8_scaled:
            args.fp8_base = True
        args.fp8_base_unet = False
        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

        if args.blocks_to_swap and args.cpu_offload_checkpointing:
            raise ValueError("Krea 2 does not combine --blocks_to_swap with --cpu_offload_checkpointing")

    def load_target_model(self, args, weight_dtype, accelerator):
        if not args.krea2_text_encoder:
            raise ValueError("--krea2_text_encoder/--text_encoder is required for Krea 2 text caching")
        if not args.vae:
            raise ValueError("--vae is required for Krea 2")

        logger.info("Loading Krea 2 Qwen3-VL text encoder on CPU")
        text_encoder = krea2_utils.load_krea2_text_encoder(
            args.krea2_text_encoder,
            dtype=torch.bfloat16,
            device="cpu",
            max_length=args.krea2_max_token_length,
            tokenizer_repo=args.krea2_tokenizer,
        )

        logger.info("Loading Qwen-Image VAE")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            input_channels=3,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        vae.to(weight_dtype).eval()
        return "krea2", [text_encoder], vae, None

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        self.is_swapping_blocks = bool(args.blocks_to_swap and args.blocks_to_swap > 0)
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device
        attn_mode = args.attn_mode or ("xformers" if args.xformers else "torch")
        if attn_mode == "sdpa":
            attn_mode = "torch"

        logger.info(
            "Loading Krea 2 DiT with attn_mode=%s, fp8_scaled=%s, loading_device=%s",
            attn_mode,
            args.fp8_scaled,
            loading_device,
        )
        model = krea2_utils.load_krea2_dit(
            args.pretrained_model_name_or_path,
            device=accelerator.device,
            dtype=weight_dtype,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
        )
        if self.is_swapping_blocks:
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        return model, text_encoders

    def prepare_unet_with_accelerator(self, args: argparse.Namespace, accelerator: Accelerator, unet):
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)
        model = accelerator.prepare(unet, device_placement=[False])
        model = accelerator.unwrap_model(model)
        model.move_to_device_except_swap_blocks(accelerator.device)
        model.prepare_block_swap_before_forward()
        return model

    def get_tokenize_strategy(self, args):
        return Krea2TokenizeStrategy(args.krea2_tokenizer, args.krea2_max_token_length)

    def get_tokenizers(self, tokenize_strategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return Krea2LatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return Krea2TextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if not args.cache_text_encoder_outputs:
            return None
        return Krea2TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            args.krea2_max_token_length,
        )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return None if args.cache_text_encoder_outputs else text_encoders

    def is_train_text_encoder(self, args):
        return False

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False] * len(text_encoders)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            logger.info("Caching Krea 2 Qwen3-VL outputs")
            text_encoders[0].to(accelerator.device)
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()
        else:
            if args.krea2_dynamic_text_encoder_cpu:
                logger.warning("Krea 2 dynamic caption encoding runs Qwen3-VL on CPU; training will be slower")
                text_encoders[0].to("cpu")
            else:
                logger.warning("Krea 2 dynamic caption encoding keeps Qwen3-VL on the training device")
                text_encoders[0].to(accelerator.device)

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ):
        if args.train_inpainting:
            raise ValueError("Krea 2 does not support train_inpainting")

        # The shared trainer otherwise encodes masked_images with the CPU-resident VAE.
        # Krea2 masked loss only needs conditioning_images/alpha_masks as a loss mask.
        original_masks = batch.get("masks")
        if original_masks is not None:
            batch["masks"] = None
        try:
            return super().process_batch(
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                accelerator,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train,
                train_text_encoder,
                train_unet,
            )
        finally:
            if original_masks is not None:
                batch["masks"] = original_masks

    def cast_unet(self, args):
        # The loader already applies bf16 or scaled FP8. The base loop must not
        # recast scaled-FP8 parameters back to bf16.
        return False

    def cast_text_encoder(self, args):
        return False

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return train_util.apply_immiscible_image_scale(args, latents)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=args.discrete_flow_shift,
        )

    @staticmethod
    def _sample_timesteps(args, latents, device, alpha_masks=None):
        height, width = latents.shape[-2:]
        batch_size = latents.shape[0]
        sampling = args.timestep_sampling

        if sampling == "sigma":
            sample = flux_train_utils.compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            ).to(device=device, dtype=torch.float32)
        elif sampling == "uniform":
            sample = torch.rand(batch_size, device=device, dtype=torch.float32)
        else:
            sample = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32) * args.sigmoid_scale)

        if sampling == "krea2_shift":
            seq_len = (height // 2) * (width // 2)
            mu = flux_train_utils.get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)(seq_len)
            sample = flux_train_utils.time_shift(mu, 1.0, sample)
        elif sampling == "shift":
            sample = flux_train_utils.time_shift(args.discrete_flow_shift, 1.0, sample)
        elif sampling == "autoshift":
            if alpha_masks is None:
                raise ValueError("autoshift timestep sampling requires alpha masks")
            shift = flux_train_utils.compute_autoshift_mask_flow_shift(alpha_masks, latents.device)
            sample = (sample * shift) / (1.0 + (shift - 1.0) * sample)
        elif sampling == "autoshift_wavelet":
            if alpha_masks is None:
                raise ValueError("autoshift_wavelet timestep sampling requires alpha masks")
            shift = flux_train_utils.compute_autoshift_wavelet_flow_shift(latents, alpha_masks)
            sample = (sample * shift) / (1.0 + (shift - 1.0) * sample)
        elif sampling == "flux_shift":
            seq_len = (height // 2) * (width // 2)
            mu = flux_train_utils.get_lin_function(y1=0.5, y2=1.15)(seq_len)
            sample = flux_train_utils.time_shift(mu, 1.0, sample)
        elif sampling == "plora":
            alpha = getattr(args, "p_lora_alpha", 1.0)
            u = torch.rand(batch_size, device=device, dtype=torch.float32)
            if getattr(args, "p_lora_bias", "left") == "left":
                sample = 1.0 - torch.pow(u, 1.0 / (alpha + 1.0))
            else:
                sample = torch.pow(u, 1.0 / (alpha + 1.0))
        elif sampling not in ("sigma", "uniform", "sigmoid"):
            raise ValueError(f"Unsupported Krea 2 timestep sampling: {sampling}")

        return sample

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        if latents.ndim == 5:
            if latents.shape[2] != 1:
                raise ValueError(f"Krea 2 expects single-frame latents, got {tuple(latents.shape)}")
            latents = latents.squeeze(2)

        latents = latents.to(accelerator.device, dtype=weight_dtype)
        noise = train_util.sample_training_noise(args, latents)
        t = self._sample_timesteps(args, latents, accelerator.device, batch.get("alpha_masks"))
        self.current_noise = noise
        self.current_sigmas = t
        noisy = latents.lerp(noise, t[:, None, None, None])
        self.current_noisy_latents = noisy
        timesteps = t * 1000.0 + 1.0

        prompt_embeds, attn_mask = text_encoder_conds[:2]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device, dtype=torch.bool)
        if args.gradient_checkpointing:
            noisy.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        def forward_model(model_input, context, context_mask):
            img, pos, mask = krea2_sampling.prepare(
                model_input, context.shape[1], unet.config.patch, context_mask
            )
            return unet(
                img=img.to(dtype=weight_dtype),
                context=context,
                t=(timesteps / 1000.0).to(device=accelerator.device),
                pos=pos,
                mask=mask,
            )

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            pred = forward_model(noisy, prompt_embeds, attn_mask)

        height = noisy.shape[-2] // unet.config.patch
        width = noisy.shape[-1] // unet.config.patch
        pred = pred.reshape(pred.shape[0], height, width, unet.config.patch, unet.config.patch, unet.config.channels)
        pred = pred.permute(0, 5, 1, 3, 2, 4).reshape_as(noisy)

        apply_ciop = bool(
            is_train
            and getattr(args, "ciop_prob", 0.0) > 0.0
            and torch.rand(1, device=latents.device).item() < args.ciop_prob
        )
        eps_out = None
        if apply_ciop:
            std = args.ciop_noise_magnitude
            if getattr(args, "ciop_noise_type", "gaussian") == "uniform":
                limit = math.sqrt(3.0) * std
                eps_in = (torch.rand_like(noisy) * 2 - 1.0) * limit
                eps_out = (torch.rand_like(latents) * 2 - 1.0) * limit
            else:
                eps_in = torch.randn_like(noisy) * std
                eps_out = torch.randn_like(latents) * std
            noisy = noisy + eps_in
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                pred = forward_model(noisy, prompt_embeds, attn_mask)
            pred = pred.reshape(pred.shape[0], height, width, unet.config.patch, unet.config.patch, unet.config.channels)
            pred = pred.permute(0, 5, 1, 3, 2, 4).reshape_as(noisy)
            self.current_noisy_latents = noisy

        target = noise - latents

        model_guidance_weight = getattr(args, "model_guidance_weight", 0.0)
        model_guidance_end = getattr(args, "model_guidance_end_step", 0)
        apply_model_guidance = model_guidance_weight > 0.0 and (
            model_guidance_end == 0 or self.global_step < model_guidance_end
        )
        if apply_model_guidance:
            if getattr(args, "model_guidance_prob", 1.0) < 1.0:
                apply_model_guidance = torch.rand(1, device=latents.device).item() <= args.model_guidance_prob

        if apply_model_guidance:
            unconditional_context = torch.zeros_like(prompt_embeds)
            unconditional_mask = torch.zeros_like(attn_mask)
            unconditional_mask[:, 0] = True
            with torch.no_grad(), accelerator.autocast():
                pred_uncond = forward_model(noisy, unconditional_context, unconditional_mask)
            pred_uncond = pred_uncond.reshape(
                pred_uncond.shape[0], height, width, unet.config.patch, unet.config.patch, unet.config.channels
            )
            pred_uncond = pred_uncond.permute(0, 5, 1, 3, 2, 4).reshape_as(noisy)

            current_weight = model_guidance_weight
            warmup_steps = getattr(args, "model_guidance_warmup_steps", 0)
            if warmup_steps > 0 and self.global_step < warmup_steps:
                current_weight *= self.global_step / warmup_steps

            if getattr(args, "model_guidance_cfg_zero", False):
                dot = torch.sum(pred * pred_uncond, dim=[1, 2, 3], keepdim=True)
                norm = torch.sum(pred_uncond**2, dim=[1, 2, 3], keepdim=True) + 1e-8
                guidance = current_weight * (dot / norm).detach()
                threshold = getattr(args, "model_guidance_zero_init_threshold", 0.95)
                guidance = guidance * (timesteps < threshold).float().view(-1, 1, 1, 1)
            elif getattr(args, "model_guidance_timestep_scaling", False):
                min_weight = getattr(args, "model_guidance_min_weight", 0.0)
                guidance = min_weight + (current_weight - min_weight) * 4.0 * timesteps * (1.0 - timesteps)
                guidance = guidance.view(-1, 1, 1, 1)
            else:
                guidance = current_weight
            target = target + guidance * (pred - pred_uncond).detach()

        if apply_ciop:
            target = target + eps_out

        target = anima_train_utils.apply_differential_guidance_target(
            target, pred, getattr(args, "differential_guidance_scale", 1.0)
        )
        weighting = anima_train_utils.compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=t, args=args
        )
        return pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        if args.sample_prompts:
            logger.warning("Krea 2 sample preview is not enabled in the shared adapter yet; training continues without samples")

    def update_metadata(self, metadata, args):
        metadata["ss_architecture"] = "krea2"
        metadata["ss_krea2_timestep_sampling"] = args.timestep_sampling
        metadata["ss_krea2_max_token_length"] = args.krea2_max_token_length


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument("--dit", dest="pretrained_model_name_or_path", type=str, help="Krea 2 RAW DiT checkpoint")
    parser.add_argument(
        "--krea2_text_encoder",
        "--text_encoder",
        dest="krea2_text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B-Instruct safetensors checkpoint",
    )
    parser.add_argument(
        "--krea2_tokenizer",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen3-VL tokenizer repository or local directory",
    )
    parser.add_argument(
        "--krea2_max_token_length",
        type=int,
        default=512,
        help="Maximum Krea 2 text cache length",
    )
    parser.add_argument(
        "--krea2_dynamic_text_encoder",
        action="store_true",
        help="Encode captions during training instead of using cached text encoder outputs",
    )
    parser.add_argument(
        "--krea2_dynamic_text_encoder_cpu",
        action="store_true",
        help="Keep Qwen3-VL on CPU during dynamic training-time caption encoding; slower but uses less VRAM",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="Load Krea 2 main blocks using dynamic scaled FP8",
    )

    timestep_action = parser._option_string_actions["--timestep_sampling"]
    if timestep_action.choices is not None and "krea2_shift" not in timestep_action.choices:
        timestep_action.choices = list(timestep_action.choices) + ["krea2_shift"]
    timestep_action.default = "krea2_shift"
    parser._option_string_actions["--discrete_flow_shift"].default = 2.5
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    if args.network_module is None:
        args.network_module = "networks.lora_krea2"
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    Krea2NetworkTrainer().train(args)
