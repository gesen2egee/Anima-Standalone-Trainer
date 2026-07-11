"""Krea 2 inference using the same model/cache conventions as training."""

import argparse
import gc
import importlib
import logging
import os
import random
from datetime import datetime

import torch
from safetensors.torch import load_file

from library import qwen_image_autoencoder_kl
from library.krea2 import krea2_sampling, krea2_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with Krea 2.")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--from_file", type=str)
    parser.add_argument("--output_type", default="images", help="Compatibility option used by Training UI")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--network_module", default="networks.lora_krea2")
    parser.add_argument("--krea2_tokenizer", default=krea2_utils.QWEN3_VL_4B_INSTRUCT_REPO_ID)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--y1", type=float, default=0.5)
    parser.add_argument("--y2", type=float, default=1.15)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--attn_mode", default="torch", choices=["torch", "flash", "sageattn", "xformers"])
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--lora_weight", nargs="*", default=None)
    parser.add_argument("--lora_multiplier", nargs="*", type=float, default=None)
    args = parser.parse_args()
    if not args.prompt and not args.from_file:
        parser.error("請提供 prompt 或 --from_file")
    if args.prompt and args.from_file:
        parser.error("prompt 與 --from_file 只能擇一")
    return args


def load_pipeline(args):
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = krea2_utils.load_krea2_text_encoder(
        args.text_encoder,
        dtype=dtype,
        device="cpu",
        tokenizer_repo=args.krea2_tokenizer,
    )
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        input_channels=3,
        device="cpu",
        disable_mmap=True,
        disable_cache=True,
    ).to(dtype=dtype).eval()
    generic_lora_module = args.network_module in {"networks.cdka", "networks.krona"}
    lora_weights = [load_file(path) for path in args.lora_weight] if args.lora_weight and not generic_lora_module else None
    dit = krea2_utils.load_krea2_dit(
        args.dit,
        device=device,
        dtype=dtype,
        fp8_scaled=args.fp8_scaled,
        loading_device="cpu" if args.blocks_to_swap else device,
        attn_mode=args.attn_mode,
        split_attn=args.split_attn,
        lora_weights=lora_weights,
        lora_multipliers=args.lora_multiplier,
    ).eval().requires_grad_(False)

    if args.lora_weight and generic_lora_module:
        network_module = importlib.import_module(args.network_module)
        multipliers = args.lora_multiplier or []
        for index, path in enumerate(args.lora_weight):
            multiplier = multipliers[index] if index < len(multipliers) else 1.0
            network, weights_sd = network_module.create_network_from_weights(
                multiplier, path, None, [None], dit, for_inference=True
            )
            merge_device = torch.device("cpu") if args.blocks_to_swap else device
            network.merge_to([None], dit, weights_sd, dtype=dtype, device=merge_device)
            del network
    if args.blocks_to_swap:
        dit.enable_block_swap(args.blocks_to_swap, device)
        dit.move_to_device_except_swap_blocks(device)
        dit.switch_block_swap_for_inference()
    return device, dtype, encoder, vae, dit


@torch.no_grad()
def generate_one(args, prompt, device, dtype, encoder, vae, dit):
    encoder = encoder.to(device)
    txt, txt_mask = encoder([prompt])
    txt, txt_mask = krea2_sampling.gather_valid_text(txt, txt_mask)
    if args.guidance_scale > 1.0:
        untxt, untxt_mask = encoder([""])
        untxt, untxt_mask = krea2_sampling.gather_valid_text(untxt, untxt_mask)
    else:
        untxt = untxt_mask = None
    encoder.to("cpu")
    if args.blocks_to_swap:
        dit.prepare_block_swap_before_forward()
    return krea2_sampling.sample(
        dit,
        vae,
        txt,
        txt_mask,
        untxt=untxt,
        untxtmask=untxt_mask,
        device=device,
        dtype=dtype,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.guidance_scale,
        seed=args.seed,
        y1=args.y1,
        y2=args.y2,
        mu=args.mu,
    )


def main():
    args = parse_args()
    device, dtype, encoder, vae, dit = load_pipeline(args)
    prompts = [args.prompt]
    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    os.makedirs(args.save_path, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    for index, prompt in enumerate(prompts):
        args.seed = seed + index
        logger.info("[%s] %s", index, prompt)
        images = generate_one(args, prompt, device, dtype, encoder, vae, dit)
        for image_index, image in enumerate(images):
            filename = f"{datetime.now():%Y%m%d-%H%M%S-%f}_{args.seed + image_index}.png"
            image.save(os.path.join(args.save_path, filename))
    del encoder, vae, dit
    gc.collect()


if __name__ == "__main__":
    main()
