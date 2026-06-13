"""UI-compatible Anima generation entry point.

This script keeps the historical anima_gen.py command name while delegating to
the sd-scripts NEW inference backend in anima_minimal_inference.py.
"""

import argparse
import sys


def _append_flag(argv: list[str], flag: str, value) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _translate_legacy_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for anima_minimal_inference.py",
        add_help=False,
    )
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--dit_path")
    parser.add_argument("--qwen3_path")
    parser.add_argument("--vae_path")
    parser.add_argument("--sample_prompts")
    parser.add_argument("--output_dir")
    parser.add_argument("--output_name")
    parser.add_argument("--mixed_precision")
    parser.add_argument("--network_weights")
    parser.add_argument("--network_mul", type=float)
    parser.add_argument("--discrete_flow_shift", type=float)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--sage_attn", action="store_true")
    parser.add_argument("--device_map")
    parser.add_argument("--server_port", type=int)

    args, passthrough = parser.parse_known_args(argv)

    if args.help:
        print(
            "usage: anima_gen.py [NEW inference args] [legacy UI aliases]\n\n"
            "Legacy UI aliases:\n"
            "  --dit_path PATH              alias for --dit\n"
            "  --qwen3_path PATH            alias for --text_encoder\n"
            "  --vae_path PATH              alias for --vae\n"
            "  --sample_prompts PATH        alias for --from_file\n"
            "  --output_dir PATH            alias for --save_path\n"
            "  --network_weights PATH       alias for --lora_weight\n"
            "  --network_mul FLOAT          alias for --lora_multiplier\n"
            "  --discrete_flow_shift FLOAT  alias for --flow_shift\n"
            "  --flash_attn                 alias for --attn_mode flash\n"
            "  --sage_attn                  alias for --attn_mode sageattn\n\n"
            "All other arguments are passed through to anima_minimal_inference.py."
        )
        return []

    if args.server_port is not None:
        raise SystemExit("anima_gen.py now uses the sd-scripts NEW backend; server mode is not supported.")
    if args.device_map:
        raise SystemExit("anima_gen.py now uses the sd-scripts NEW backend; device_map is not supported.")

    translated = list(passthrough)
    _append_flag(translated, "--dit", args.dit_path)
    _append_flag(translated, "--text_encoder", args.qwen3_path)
    _append_flag(translated, "--vae", args.vae_path)
    _append_flag(translated, "--from_file", args.sample_prompts)
    _append_flag(translated, "--save_path", args.output_dir)
    _append_flag(translated, "--lora_weight", args.network_weights)
    _append_flag(translated, "--lora_multiplier", args.network_mul)
    _append_flag(translated, "--flow_shift", args.discrete_flow_shift)

    if args.flash_attn:
        translated.extend(["--attn_mode", "flash"])
    elif args.sage_attn:
        translated.extend(["--attn_mode", "sageattn"])

    return translated


def main() -> None:
    translated = _translate_legacy_args(sys.argv[1:])
    if not translated:
        return

    from anima_minimal_inference import main as minimal_main

    sys.argv = ["anima_minimal_inference.py", *translated]
    minimal_main()


if __name__ == "__main__":
    main()
