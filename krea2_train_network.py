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

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
        if not args.cache_text_encoder_outputs:
            raise ValueError(
                "Krea 2 requires --cache_text_encoder_outputs. "
                "Run the text-cache stage before starting training."
            )
        if not args.cache_latents:
            raise ValueError(
                "Krea 2 requires --cache_latents because the shared trainer keeps the Qwen-Image VAE on CPU."
            )
        if not train_dataset_group.is_text_encoder_output_cacheable(False):
            raise ValueError(
                "Krea 2 text cache does not support caption dropout, caption shuffle, token warmup, or tag dropout."
            )
        if val_dataset_group is not None and not val_dataset_group.is_text_encoder_output_cacheable(False):
            raise ValueError("Krea 2 validation text cache has unsupported caption augmentation settings.")
        if args.network_train_text_encoder_only:
            raise ValueError("Krea 2 text encoder is frozen; --network_train_text_encoder_only is not supported")

        args.network_train_unet_only = True
        args.fp8_base = False
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
        return Krea2TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            args.krea2_max_token_length,
        )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return None

    def is_train_text_encoder(self, args):
        return False

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False] * len(text_encoders)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        logger.info("Caching Krea 2 Qwen3-VL outputs")
        text_encoders[0].to(accelerator.device)
        dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
        text_encoders[0].to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

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
    def _sample_timesteps(args, latents, device):
        height, width = latents.shape[-2:]
        sample = torch.rand(latents.shape[0], device=device, dtype=torch.float32)
        sampling = args.timestep_sampling

        if sampling in ("sigmoid", "krea2_shift", "shift"):
            sample = torch.sigmoid(torch.erfinv(sample.mul(2.0).sub(1.0)) * math.sqrt(2.0) * args.sigmoid_scale)

        if sampling == "krea2_shift":
            seq_len = (height // 2) * (width // 2)
            mu = flux_train_utils.get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)(seq_len)
            shift = math.exp(mu)
            sample = (sample * shift) / (1.0 + (shift - 1.0) * sample)
        elif sampling == "shift":
            shift = math.exp(args.discrete_flow_shift)
            sample = (sample * shift) / (1.0 + (shift - 1.0) * sample)
        elif sampling not in ("uniform", "sigmoid"):
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
        t = self._sample_timesteps(args, latents, accelerator.device)
        noisy = latents.lerp(noise, t[:, None, None, None])
        timesteps = t * 1000.0 + 1.0

        prompt_embeds, attn_mask = text_encoder_conds[:2]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device, dtype=torch.bool)
        img, pos, mask = krea2_sampling.prepare(noisy, prompt_embeds.shape[1], unet.config.patch, attn_mask)

        if args.gradient_checkpointing:
            img.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            pred = unet(
                img=img.to(dtype=weight_dtype),
                context=prompt_embeds,
                t=(timesteps / 1000.0).to(device=accelerator.device),
                pos=pos,
                mask=mask,
            )

        height = noisy.shape[-2] // unet.config.patch
        width = noisy.shape[-1] // unet.config.patch
        pred = pred.reshape(pred.shape[0], height, width, unet.config.patch, unet.config.patch, unet.config.channels)
        pred = pred.permute(0, 5, 1, 3, 2, 4).reshape_as(noisy)
        target = noise - latents
        return pred, target, timesteps, None

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
