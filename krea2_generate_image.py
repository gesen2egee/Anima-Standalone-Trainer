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


def parse_prompt_line(line: str) -> dict:
    """Parse the prompt-file syntax shared with the Training UI/sd-scripts."""
    parts = line.strip().split(" --")
    spec = {"prompt": parts[0].strip()}
    converters = {
        "w": ("width", int),
        "h": ("height", int),
        "s": ("steps", int),
        "d": ("seed", int),
        "l": ("guidance_scale", float),
        "g": ("guidance_scale", float),
        "n": ("negative_prompt", str),
    }
    for raw_arg in parts[1:]:
        name, separator, value = raw_arg.partition(" ")
        if not separator or name.lower() not in converters:
            continue
        key, converter = converters[name.lower()]
        try:
            spec[key] = converter(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid --{name} value in prompt line: {value!r}") from exc
    if not spec["prompt"]:
        raise ValueError("Prompt text cannot be empty")
    return spec


def load_prompt_specs(prompt: str | None, from_file: str | None) -> list[dict]:
    if prompt is not None:
        return [{"prompt": prompt}]
    with open(from_file, "r", encoding="utf-8") as file:
        specs = [
            parse_prompt_line(line)
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not specs:
        raise ValueError(f"No usable prompts found in {from_file}")
    return specs


def get_runtime_device_dtype():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.bfloat16


def load_text_encoder(args, device, dtype):
    return krea2_utils.load_krea2_text_encoder(
        args.text_encoder,
        dtype=dtype,
        device="cpu",
        tokenizer_repo=args.krea2_tokenizer,
    )


@torch.no_grad()
def preencode_prompt_specs(args, prompt_specs, encoder, device):
    """Encode all prompt branches before loading the DiT to avoid a VRAM overlap."""
    encoder.to(device)
    cache = {}
    for spec in prompt_specs:
        guidance_scale = spec.get("guidance_scale", args.guidance_scale)
        branches = [("text_condition", spec["prompt"])]
        if guidance_scale > 1.0:
            branches.append(("negative_text_condition", spec.get("negative_prompt", "")))
        for destination, text in branches:
            if text not in cache:
                hidden, mask = encoder([text])
                hidden, mask = krea2_sampling.gather_valid_text(hidden, mask)
                cache[text] = (hidden.cpu(), mask.cpu())
            spec[destination] = cache[text]
    encoder.to("cpu")
    return prompt_specs


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


def load_pipeline(args, load_encoder=True):
    device, dtype = get_runtime_device_dtype()
    encoder = load_text_encoder(args, device, dtype) if load_encoder else None
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        input_channels=3,
        device="cpu",
        disable_mmap=True,
        disable_cache=True,
    ).to(dtype=dtype).eval()
    generic_lora_module = args.network_module in {"networks.cdka", "networks.krona"}
    if args.fp8_scaled and args.lora_weight and generic_lora_module:
        raise ValueError(
            f"{args.network_module} must be merged before FP8 quantization; disable --fp8_scaled for this adapter"
        )
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
def generate_one(args, prompt_spec, device, dtype, encoder, vae, dit):
    prompt = prompt_spec["prompt"] if isinstance(prompt_spec, dict) else str(prompt_spec)
    negative_prompt = prompt_spec.get("negative_prompt", "") if isinstance(prompt_spec, dict) else ""
    width = prompt_spec.get("width", args.width) if isinstance(prompt_spec, dict) else args.width
    height = prompt_spec.get("height", args.height) if isinstance(prompt_spec, dict) else args.height
    steps = prompt_spec.get("steps", args.steps) if isinstance(prompt_spec, dict) else args.steps
    guidance_scale = (
        prompt_spec.get("guidance_scale", args.guidance_scale)
        if isinstance(prompt_spec, dict)
        else args.guidance_scale
    )
    seed = prompt_spec.get("seed", args.seed) if isinstance(prompt_spec, dict) else args.seed
    if width <= 0 or height <= 0 or steps <= 0 or guidance_scale < 0:
        raise ValueError(
            f"Invalid generation settings: width={width}, height={height}, steps={steps}, guidance={guidance_scale}"
        )

    if isinstance(prompt_spec, dict) and "text_condition" in prompt_spec:
        txt, txt_mask = prompt_spec["text_condition"]
        if guidance_scale > 1.0:
            untxt, untxt_mask = prompt_spec["negative_text_condition"]
        else:
            untxt = untxt_mask = None
    else:
        if encoder is None:
            raise RuntimeError("Krea 2 prompt was not pre-encoded and no Text Encoder is loaded")
        encoder = encoder.to(device)
        txt, txt_mask = encoder([prompt])
        txt, txt_mask = krea2_sampling.gather_valid_text(txt, txt_mask)
        if guidance_scale > 1.0:
            untxt, untxt_mask = encoder([negative_prompt])
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
        width=width,
        height=height,
        steps=steps,
        cfg_scale=guidance_scale,
        seed=seed,
        y1=args.y1,
        y2=args.y2,
        mu=args.mu,
    )


def main():
    args = parse_args()
    prompt_specs = load_prompt_specs(args.prompt, args.from_file)
    os.makedirs(args.save_path, exist_ok=True)
    device, dtype = get_runtime_device_dtype()
    encoder = load_text_encoder(args, device, dtype)
    preencode_prompt_specs(args, prompt_specs, encoder, device)
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device, dtype, encoder, vae, dit = load_pipeline(args, load_encoder=False)
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    for index, prompt_spec in enumerate(prompt_specs):
        prompt_spec.setdefault("seed", base_seed + index)
        logger.info("[%s] %s", index, prompt_spec["prompt"])
        images = generate_one(args, prompt_spec, device, dtype, encoder, vae, dit)
        for image_index, image in enumerate(images):
            filename = f"{datetime.now():%Y%m%d-%H%M%S-%f}_{prompt_spec['seed'] + image_index}.png"
            image.save(os.path.join(args.save_path, filename))
    del vae, dit
    gc.collect()


if __name__ == "__main__":
    main()
