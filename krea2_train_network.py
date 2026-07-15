"""Krea 2 LoRA training through the existing Anima/sd-scripts pipeline."""

import argparse
import logging
import math
import os
import time
import torch
from accelerate import Accelerator

import train_network
from library import anima_train_utils, flow_network_trainer, flux_train_utils, qwen_image_autoencoder_kl, self_flow, train_util
from library.device_utils import clean_memory_on_device
from library.krea2 import krea2_sampling, krea2_utils
from library.strategy_krea2 import (
    Krea2LatentsCachingStrategy,
    Krea2TextEncoderOutputsCachingStrategy,
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
)

logger = logging.getLogger(__name__)


class Krea2NetworkTrainer(flow_network_trainer.FlowNetworkTrainerMixin, train_network.NetworkTrainer):
    """Architecture adapter that keeps the shared optimizer/dataset/checkpoint loop."""

    cdc_architecture_name = "krea2"

    def __init__(self):
        super().__init__()
        self.is_swapping_blocks = False
        self._unconditional_text_encoder_conds = None
        self._sample_prompt_conds = {}

    @staticmethod
    def _unpatchify_prediction(pred: torch.Tensor, reference: torch.Tensor, patch: int, channels: int) -> torch.Tensor:
        """Restore Krea 2 ``(c ph pw)`` image tokens to a channel-first latent."""
        height = reference.shape[-2] // patch
        width = reference.shape[-1] // patch
        expected_shape = (reference.shape[0], height * width, channels * patch * patch)
        if tuple(pred.shape) != expected_shape:
            raise ValueError(f"Krea 2 prediction shape must be {expected_shape}, got {tuple(pred.shape)}")
        pred = pred.reshape(pred.shape[0], height, width, channels, patch, patch)
        return pred.permute(0, 3, 1, 4, 2, 5).reshape_as(reference)

    @staticmethod
    def _dataset_needs_dynamic_caption_encoding(dataset) -> bool:
        """Return whether caption processing changes between training samples/steps."""
        datasets = getattr(dataset, "datasets", None)
        if datasets is None:
            datasets = [dataset]
        for child_dataset in datasets:
            for subset in getattr(child_dataset, "subsets", []):
                if Krea2NetworkTrainer._subset_needs_dynamic_caption_encoding(subset):
                    return True
        return False

    @staticmethod
    def _subset_needs_dynamic_caption_encoding(subset) -> bool:
        return bool(
            subset.caption_dropout_rate > 0
            or subset.caption_dropout_every_n_epochs > 0
            or subset.shuffle_caption
            or subset.token_warmup_step > 0
            or subset.caption_tag_dropout_rate > 0
            or getattr(subset, "enable_fad", False)
            or subset.caption_prefix
            or subset.caption_suffix
            or subset.enable_wildcard
        )

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        self.validate_flow_training_args(args)
        train_text_encoder = not args.network_train_unet_only or args.network_train_text_encoder_only
        if train_text_encoder:
            args.krea2_dynamic_text_encoder = True
            args.krea2_dynamic_text_encoder_cpu = False
            if args.krea2_text_encoder_layer_offload:
                logger.warning(
                    "Krea 2 Text Encoder adapter training needs backward activations; disabling forward-only TE Layer Offload."
                )
                args.krea2_text_encoder_layer_offload = False
        if args.krea2_max_token_length <= 0:
            raise ValueError("krea2_max_token_length must be greater than 0")
        if args.blocks_to_swap is not None and not 0 <= args.blocks_to_swap <= 26:
            raise ValueError("Krea 2 blocks_to_swap must be between 0 and 26")
        if args.attn_mode == "sageattn":
            raise ValueError("Krea 2 SageAttention is inference-only; use torch, xformers, or flash for training")
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
        dynamic_caption_encoding = self._dataset_needs_dynamic_caption_encoding(train_dataset_group)
        if val_dataset_group is not None:
            dynamic_caption_encoding = dynamic_caption_encoding or self._dataset_needs_dynamic_caption_encoding(val_dataset_group)

        dynamic_text_encoder = (
            getattr(args, "krea2_dynamic_text_encoder", False)
            or getattr(args, "krea2_dynamic_text_encoder_cpu", False)
        )

        if (
            dynamic_caption_encoding
            and not dynamic_text_encoder
            and not getattr(args, "krea2_text_encoder_layer_offload", False)
        ):
            logger.warning(
                "Krea 2 caption augmentation requires dynamic Qwen3-VL encoding; "
                "enabling TE Layer Offload to avoid overlapping the full Text Encoder with the DiT."
            )
            args.krea2_text_encoder_layer_offload = True

        if args.krea2_text_encoder_layer_offload and not 0.0 < args.krea2_text_encoder_offload_percent <= 1.0:
            raise ValueError("krea2_text_encoder_offload_percent must be in (0, 1]")

        if getattr(args, "krea2_dynamic_text_encoder_cpu", False) and getattr(
            args, "krea2_text_encoder_layer_offload", False
        ):
            raise ValueError("Krea 2 cannot combine whole-CPU TE encoding with TE Layer Offload")

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
        if args.train_inpainting:
            raise ValueError("Krea 2 supports masked loss, but the Anima inpainting input path is not compatible")

        # Krea 2 only supports the safe scaled-FP8 path. Keep the shared
        # fp8_base flag as the compatibility gate, but reject plain FP8 rather
        # than silently running in a different precision than the UI requests.
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Krea 2 requires FP8 Scaled together with FP8 Base; plain FP8 is not supported.")
        if args.fp8_scaled:
            # The Krea2 loader already quantizes only the supported Linear
            # layers. Leaving the shared flag enabled would make the generic
            # trainer recast every DiT parameter to raw float8 afterward.
            args.fp8_base = False
        args.fp8_base_unet = False
        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

        if args.cpu_offload_checkpointing or getattr(args, "unsloth_offload_checkpointing", False):
            raise ValueError(
                "Krea 2 activation offload is not implemented; use gradient checkpointing with --blocks_to_swap instead"
            )

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
        bypass_weights = None
        if args.krea2_bypass:
            logger.info("Krea 2 bypass enabled; merging TextFusion adapter before training and sampling")
            bypass_weights = [krea2_utils.load_krea2_bypass_lora(args.krea2_bypass_lora)]
        model = krea2_utils.load_krea2_dit(
            args.pretrained_model_name_or_path,
            device=accelerator.device,
            dtype=weight_dtype,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
            lora_weights=bypass_weights,
            lora_multipliers=[1.0] if bypass_weights else None,
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
        return not args.network_train_unet_only or args.network_train_text_encoder_only

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.is_train_text_encoder(args)] * len(text_encoders)

    def is_text_encoder_not_needed_for_training(self, args):
        return bool(args.cache_text_encoder_outputs)

    @staticmethod
    def _needs_unconditional_conditioning(args) -> bool:
        return float(getattr(args, "model_guidance_weight", 0.0) or 0.0) > 0.0

    def _encode_empty_prompt_conditioning(self, args, text_encoder, accelerator, refresh=False):
        if (self._unconditional_text_encoder_conds is not None and not refresh) or not self._needs_unconditional_conditioning(args):
            return

        logger.info("Encoding Krea 2 empty-prompt conditioning for Model Guidance")
        with torch.no_grad():
            empty_hidden, empty_mask = krea2_utils.get_krea2_prompt_embeds(text_encoder, [""])
            empty_hidden, empty_mask = krea2_sampling.gather_valid_text(empty_hidden, empty_mask.bool())
        self._unconditional_text_encoder_conds = (empty_hidden.cpu(), empty_mask.cpu())

    def _cache_sample_prompt_conditioning(self, args, text_encoder, accelerator):
        if not accelerator.is_main_process or not args.sample_prompts or not os.path.isfile(args.sample_prompts):
            return
        prompts = train_util.load_prompts(args.sample_prompts)
        texts = []
        for prompt_dict in prompts:
            texts.append(prompt_dict.get("prompt", ""))
            if float(prompt_dict.get("guidance_scale", prompt_dict.get("scale", 5.5))) > 1.0:
                texts.append(prompt_dict.get("negative_prompt", ""))
        texts = list(dict.fromkeys(texts))
        if not texts:
            return
        logger.info("Caching Krea 2 sample prompt conditioning (%d unique prompts)", len(texts))
        with torch.no_grad():
            for text in texts:
                hidden, mask = krea2_utils.get_krea2_prompt_embeds(text_encoder, [text])
                hidden, mask = krea2_sampling.gather_valid_text(hidden, mask.bool())
                self._sample_prompt_conds[text] = (hidden.cpu(), mask.cpu())

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        text_encoders[0].set_gradient_enabled(self.is_train_text_encoder(args))
        if args.cache_text_encoder_outputs:
            logger.info("Caching Krea 2 Qwen3-VL outputs")
            if getattr(args, "krea2_text_encoder_layer_offload", False):
                text_encoders[0].enable_layer_offload(
                    accelerator.device, getattr(args, "krea2_text_encoder_offload_percent", 1.0)
                )
            else:
                text_encoders[0].to(accelerator.device)
            self._encode_empty_prompt_conditioning(args, text_encoders[0], accelerator)
            self._cache_sample_prompt_conditioning(args, text_encoders[0], accelerator)
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
            if getattr(args, "krea2_text_encoder_layer_offload", False):
                text_encoders[0].disable_layer_offload()
            else:
                text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()
        else:
            if getattr(args, "krea2_text_encoder_layer_offload", False):
                logger.info("Krea 2 dynamic caption encoding uses AIT-style Qwen3-VL Layer Offload")
                text_encoders[0].enable_layer_offload(
                    accelerator.device, getattr(args, "krea2_text_encoder_offload_percent", 1.0)
                )
                self._encode_empty_prompt_conditioning(args, text_encoders[0], accelerator)
                self._cache_sample_prompt_conditioning(args, text_encoders[0], accelerator)
            elif args.krea2_dynamic_text_encoder_cpu:
                logger.warning("Krea 2 dynamic caption encoding runs Qwen3-VL on CPU; training will be slower")
                # The one-time Model Guidance condition is much faster on GPU.
                # Return the frozen TE to CPU immediately afterward.
                if self._unconditional_text_encoder_conds is None and self._needs_unconditional_conditioning(args):
                    text_encoders[0].to(accelerator.device)
                    self._encode_empty_prompt_conditioning(args, text_encoders[0], accelerator)
                    text_encoders[0].to("cpu")
                    clean_memory_on_device(accelerator.device)
                else:
                    text_encoders[0].to("cpu")
                self._cache_sample_prompt_conditioning(args, text_encoders[0], accelerator)
            else:
                logger.warning("Krea 2 dynamic caption encoding keeps Qwen3-VL on the training device")
                text_encoders[0].to(accelerator.device)
                self._encode_empty_prompt_conditioning(args, text_encoders[0], accelerator)
                self._cache_sample_prompt_conditioning(args, text_encoders[0], accelerator)

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
            if train_text_encoder and self._needs_unconditional_conditioning(args):
                # The Text Encoder adapter changes every optimizer step. Refresh
                # Model Guidance conditioning so it never uses stale TE weights.
                self._encode_empty_prompt_conditioning(args, text_encoders[0], accelerator, refresh=True)
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
        noisy, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args,
            noise_scheduler,
            latents,
            noise,
            accelerator.device,
            weight_dtype,
            batch.get("alpha_masks"),
            folder_shifts=batch.get("folder_shifts"),
            batch_timesteps=batch.get("timesteps"),
            folder_shift_progress=batch.get("folder_shift_progress"),
            automask_shift_values=batch.get("automask_shift_values"),
        )
        t = sigmas.reshape(sigmas.shape[0], -1)[:, 0].to(dtype=torch.float32)
        self.current_self_flow_representation_loss = None
        self_flow_teacher_input = None
        model_timesteps = (timesteps / 1000.0).to(device=accelerator.device)
        if is_train and getattr(args, "use_self_flow", False):
            _, _, second_sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                args,
                noise_scheduler,
                latents,
                noise,
                accelerator.device,
                weight_dtype,
                batch.get("alpha_masks"),
                folder_shifts=batch.get("folder_shifts"),
                batch_timesteps=None,
                folder_shift_progress=batch.get("folder_shift_progress"),
                automask_shift_values=batch.get("automask_shift_values"),
            )
            patch = unet.config.patch
            token_height = latents.shape[-2] // patch
            token_width = latents.shape[-1] // patch
            token_mask = self_flow.sample_dual_timestep_mask(
                latents.shape[0], token_height, token_width, args.self_flow_mask_ratio, latents.device
            )
            token_sigmas = self_flow.mix_token_timesteps(sigmas, second_sigmas, token_mask)
            latent_sigmas = self_flow.expand_token_grid(token_sigmas, patch).to(dtype=latents.dtype)
            noisy = (1.0 - latent_sigmas) * latents + latent_sigmas * noise
            teacher_sigmas = torch.minimum(
                sigmas.reshape(sigmas.shape[0]), second_sigmas.reshape(second_sigmas.shape[0])
            )
            teacher_latent_sigmas = teacher_sigmas[:, None, None, None].to(dtype=latents.dtype)
            self_flow_teacher_input = (1.0 - teacher_latent_sigmas) * latents + teacher_latent_sigmas * noise
            model_timesteps = token_sigmas.flatten(1).to(dtype=torch.float32)
        cdc_result = self.apply_cdc_flow_path(args, batch, latents, noise, sigmas, is_train)
        cdc_target = None
        if cdc_result is not None:
            noisy, cdc_target = cdc_result
        self.current_noise = noise
        self.current_sigmas = sigmas

        ciop = self.sample_ciop_perturbations(args, noisy, is_train)
        if ciop is not None:
            noisy = noisy + ciop[0]
        self.current_noisy_latents = noisy

        prompt_embeds, attn_mask = text_encoder_conds[:2]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device, dtype=torch.bool)
        if args.gradient_checkpointing:
            noisy.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        def forward_model(
            model_input,
            context,
            context_mask,
            timestep_values,
            conditioning_timestep=None,
            return_hidden_at=None,
        ):
            img, pos, mask = krea2_sampling.prepare(
                model_input, context.shape[1], unet.config.patch, context_mask
            )
            return unet(
                img=img.to(dtype=weight_dtype),
                context=context,
                t=timestep_values,
                pos=pos,
                mask=mask,
                conditioning_t=conditioning_timestep,
                return_hidden_at=return_hidden_at,
            )

        if self_flow_teacher_input is not None:
            if self.is_swapping_blocks:
                unet.prepare_block_swap_before_forward()
            with self.self_flow_teacher_context(accelerator.unwrap_model(network), unet):
                with torch.no_grad(), accelerator.autocast():
                    _, teacher_hidden = forward_model(
                        self_flow_teacher_input,
                        prompt_embeds,
                        attn_mask,
                        teacher_sigmas.to(dtype=torch.float32),
                        conditioning_timestep=teacher_sigmas.to(dtype=torch.float32),
                        return_hidden_at=args.self_flow_teacher_layer,
                    )
            if self.is_swapping_blocks:
                unet.prepare_block_swap_before_forward()
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_output = forward_model(
                noisy,
                prompt_embeds,
                attn_mask,
                model_timesteps,
                conditioning_timestep=t,
                return_hidden_at=args.self_flow_student_layer if self_flow_teacher_input is not None else None,
            )
        if self_flow_teacher_input is not None:
            pred, student_hidden = model_output
            projector = getattr(accelerator.unwrap_model(network), self_flow.PROJECTOR_MODULE_NAME)
            with accelerator.autocast():
                self.current_self_flow_representation_loss = self_flow.representation_loss(
                    projector, student_hidden, teacher_hidden
                )
        else:
            pred = model_output

        pred = self._unpatchify_prediction(pred, noisy, unet.config.patch, unet.config.channels)

        pred_uncond = None
        if self.should_apply_model_guidance(args, latents.device):
            if self._unconditional_text_encoder_conds is None:
                # This only occurs for direct unit calls that bypass trainer setup.
                # The normal training path initializes the real empty-prompt condition above.
                unconditional_context = torch.zeros_like(prompt_embeds)
                unconditional_mask = torch.zeros_like(attn_mask)
                unconditional_mask[:, 0] = True
            else:
                empty_hidden, empty_mask = self._unconditional_text_encoder_conds
                unconditional_context = empty_hidden.to(device=accelerator.device, dtype=weight_dtype).expand(
                    prompt_embeds.shape[0], -1, -1, -1
                )
                unconditional_mask = empty_mask.to(device=accelerator.device).expand(prompt_embeds.shape[0], -1)
            if self.is_swapping_blocks:
                unet.prepare_block_swap_before_forward()
            with torch.no_grad(), accelerator.autocast():
                pred_uncond = forward_model(
                    noisy,
                    unconditional_context,
                    unconditional_mask,
                    model_timesteps,
                    conditioning_timestep=t,
                )
            pred_uncond = self._unpatchify_prediction(
                pred_uncond, noisy, unet.config.patch, unet.config.channels
            )

        target, weighting = self.finalize_flow_target(
            args,
            noise,
            latents,
            pred,
            t.view(-1, 1, 1, 1),
            t,
            model_pred_uncond=pred_uncond,
            ciop_output=ciop[1] if ciop is not None else None,
            folder_shifts=batch.get("folder_shifts"),
            folder_shift_progress=batch.get("folder_shift_progress"),
            proposal_flow_shift=math.exp(
                flux_train_utils.get_krea2_resolution_shift_mu(
                    (latents.shape[-2] // 2) * (latents.shape[-1] // 2),
                    args.discrete_flow_shift,
                )
            )
            if args.timestep_sampling == "krea2_shift"
            else None,
            base_target=cdc_target,
        )
        return pred, target, timesteps, weighting

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        if not args.sample_prompts:
            return
        if global_step == 0:
            if not args.sample_at_first:
                return
        elif args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        elif args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        elif epoch is not None or global_step % args.sample_every_n_steps != 0:
            return
        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return
        if not os.path.isfile(args.sample_prompts):
            logger.error("No Krea 2 sample prompt file: %s", args.sample_prompts)
            accelerator.wait_for_everyone()
            return

        dit = accelerator.unwrap_model(unet)
        live_text_encoder = None
        if text_encoder is not None:
            live_text_encoder = text_encoder[0] if isinstance(text_encoder, (list, tuple)) else text_encoder
            live_text_encoder = accelerator.unwrap_model(live_text_encoder)
        prompts = train_util.load_prompts(args.sample_prompts)
        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Generating %d Krea 2 sample image(s) at step %d", len(prompts), global_step)

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        dit.switch_block_swap_for_inference()
        try:
            for prompt_dict in prompts:
                prompt = prompt_dict.get("prompt", "")
                negative_prompt = prompt_dict.get("negative_prompt", "")
                guidance_scale = float(prompt_dict.get("guidance_scale", prompt_dict.get("scale", 5.5)))

                def get_condition(text):
                    # Text Encoder adapters must be sampled live; frozen/deleted
                    # encoders use the CPU cache prepared before training.
                    if live_text_encoder is not None and self.is_train_text_encoder(args):
                        hidden, mask = krea2_utils.get_krea2_prompt_embeds(live_text_encoder, [text])
                        return krea2_sampling.gather_valid_text(hidden, mask.bool())
                    return self._sample_prompt_conds.get(text)

                condition = get_condition(prompt)
                negative_condition = get_condition(negative_prompt) if guidance_scale > 1.0 else None
                if condition is None or (guidance_scale > 1.0 and negative_condition is None):
                    logger.warning("Missing cached Krea 2 sample conditioning; skipping prompt: %s", prompt)
                    continue
                txt, txt_mask = condition
                if negative_condition is not None:
                    untxt, untxt_mask = negative_condition
                else:
                    untxt = untxt_mask = None

                dit.prepare_block_swap_before_forward()
                seed = prompt_dict.get("seed", args.seed if args.seed is not None else 0)
                images = krea2_sampling.sample(
                    dit,
                    vae,
                    txt,
                    txt_mask,
                    untxt=untxt,
                    untxtmask=untxt_mask,
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                    width=prompt_dict.get("width", 1024),
                    height=prompt_dict.get("height", 1024),
                    steps=prompt_dict.get("sample_steps", 28),
                    cfg_scale=guidance_scale,
                    seed=seed,
                    mu=float(prompt_dict["flow_shift"]) if "flow_shift" in prompt_dict else None,
                )
                suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
                enum = prompt_dict.get("enum", 0)
                timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                for image_index, image in enumerate(images):
                    seed_suffix = seed + image_index
                    filename = f"{args.output_name or 'krea2'}_{suffix}_{enum:02d}_{timestamp}_{seed_suffix}.png"
                    image.save(os.path.join(save_dir, filename))
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
            dit.switch_block_swap_for_training()
            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

    def update_metadata(self, metadata, args):
        self.update_flow_metadata(metadata, args)
        metadata["ss_architecture"] = "krea2"
        metadata["ss_krea2_timestep_sampling"] = args.timestep_sampling
        metadata["ss_krea2_max_token_length"] = args.krea2_max_token_length
        metadata["ss_krea2_fp8_scaled"] = bool(args.fp8_scaled)
        metadata["ss_krea2_blocks_to_swap"] = args.blocks_to_swap or 0
        metadata["ss_krea2_dynamic_text_encoder"] = not bool(args.cache_text_encoder_outputs)
        metadata["ss_krea2_text_encoder_layer_offload"] = bool(args.krea2_text_encoder_layer_offload)

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    flow_network_trainer.add_flow_network_training_arguments(parser)

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
        "--krea2_text_encoder_layer_offload",
        action="store_true",
        help="Use AIT-style Qwen3-VL Layer Offload: keep Linear weights on CPU and compute each layer on GPU",
    )
    parser.add_argument(
        "--krea2_text_encoder_offload_percent",
        type=float,
        default=1.0,
        help="Fraction of Qwen3-VL Linear layers kept on CPU for Layer Offload (0, 1]",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="Load Krea 2 main blocks using dynamic scaled FP8",
    )
    parser.add_argument(
        "--krea2_bypass",
        action="store_true",
        help="Merge the Krea 2 TextFusion refusal-reduction adapter into the base DiT before training and sampling",
    )
    parser.add_argument(
        "--krea2_bypass_lora",
        type=str,
        default=krea2_utils.DEFAULT_KREA2_BYPASS_LORA_PATH,
        help="Krea 2 TextFusion refusal-reduction LoRA path",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="Compatibility flag; Krea 2 rejects activation offload because it is not implemented",
    )

    timestep_action = parser._option_string_actions["--timestep_sampling"]
    if timestep_action.choices is not None and "krea2_shift" not in timestep_action.choices:
        timestep_action.choices = list(timestep_action.choices) + ["krea2_shift"]
    timestep_action.default = "autoshift"
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
