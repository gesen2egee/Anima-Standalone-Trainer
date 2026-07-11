"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.
"""

import logging
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from library.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use).
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


class _Qwen3VLLayerOffloader:
    """AIT-style forward-only offload for the frozen Qwen3-VL encoder.

    Non-linear modules stay on the execution device. Linear weights remain in
    pinned CPU memory and are staged to the GPU only for their own forward
    call. Krea 2 never trains this encoder, so there is no backward-side weight
    staging or optimizer state to manage.
    """

    def __init__(self, model: nn.Module, device: torch.device, offload_percent: float = 1.0):
        if device.type != "cuda":
            raise ValueError("Qwen3-VL layer offload requires a CUDA execution device")
        if not 0.0 < offload_percent <= 1.0:
            raise ValueError(f"Qwen3-VL offload percent must be in (0, 1], got {offload_percent}")

        self.model = model
        self.device = torch.device(device)
        self.offload_percent = float(offload_percent)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.managed: list[nn.Linear] = []

        self._attach()

    @staticmethod
    def _parameter_storage_key(parameter: nn.Parameter) -> tuple[int, int, tuple[int, ...]]:
        return (parameter.data_ptr(), parameter.storage_offset(), tuple(parameter.shape))

    @staticmethod
    def _move_direct_state(module: nn.Module, device: torch.device):
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(device, non_blocking=True)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                module._buffers[name] = buffer.to(device, non_blocking=True)

    @staticmethod
    def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or not torch.cuda.is_available() or tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            # Pinning is an optimization only; some Windows/CUDA combinations
            # can reject individual allocations.
            return tensor

    def _attach(self):
        linear_modules = [module for module in self.model.modules() if isinstance(module, nn.Linear)]

        # Tied weights, notably Qwen's input embedding/lm_head, must stay
        # together on the execution device. Otherwise moving the embedding
        # would silently invalidate the offloader's CPU reference.
        owners: dict[tuple[int, int, tuple[int, ...]], list[nn.Module]] = {}
        for module in self.model.modules():
            for parameter in module.parameters(recurse=False):
                owners.setdefault(self._parameter_storage_key(parameter), []).append(module)

        requested = max(1, int(round(len(linear_modules) * self.offload_percent)))
        candidate_modules = []
        for module in linear_modules:
            key = self._parameter_storage_key(module.weight)
            if any(not isinstance(owner, nn.Linear) for owner in owners.get(key, [])):
                continue
            candidate_modules.append(module)

        managed_set = set(candidate_modules[:requested])
        for module in self.model.modules():
            if module in managed_set:
                weight = module.weight.data
                if weight.device.type != "cpu":
                    weight = weight.cpu()
                module.weight.data = self._pin_cpu_tensor(weight)
                if module.bias is not None:
                    bias = module.bias.data
                    if bias.device.type != "cpu":
                        bias = bias.cpu()
                    module.bias.data = self._pin_cpu_tensor(bias)
                self._install_linear_hooks(module)
                self.managed.append(module)
            else:
                self._move_direct_state(module, self.device)

        logger.info(
            "Enabled Qwen3-VL Layer Offload: %d/%d Linear layers staged from CPU (%.0f%%)",
            len(self.managed),
            len(linear_modules),
            self.offload_percent * 100.0,
        )

    def _install_linear_hooks(self, module: nn.Linear):
        def before_forward(layer: nn.Linear, _inputs):
            cpu_weight = layer.weight.data
            cpu_bias = layer.bias.data if layer.bias is not None else None
            layer._krea2_te_cpu_state = (cpu_weight, cpu_bias)
            layer.weight.data = cpu_weight.to(self.device, non_blocking=True)
            if layer.bias is not None:
                layer.bias.data = cpu_bias.to(self.device, non_blocking=True)

        def after_forward(layer: nn.Linear, _inputs, output):
            cpu_weight, cpu_bias = layer._krea2_te_cpu_state
            layer.weight.data = cpu_weight
            if layer.bias is not None:
                layer.bias.data = cpu_bias
            del layer._krea2_te_cpu_state
            return output

        self.handles.append(module.register_forward_pre_hook(before_forward))
        self.handles.append(module.register_forward_hook(after_forward))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.to("cpu")
        self.managed.clear()


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _dequantize_comfyui_fp8_state_dict(sd: dict[str, Tensor], dtype: torch.dtype | None) -> dict[str, Tensor]:
    """Convert ComfyUI scaled-FP8 Linear weights to ordinary model weights.

    ComfyUI stores a float8 weight together with ``weight_scale`` and a JSON
    ``comfy_quant`` marker.  The regular Transformers Qwen3-VL Linear modules
    do not understand those extra tensors, so the scale must be applied before
    ``load_state_dict``.  Unknown quantization formats are rejected instead of
    silently dropping metadata.
    """
    quantized_layers = [key[: -len(".comfy_quant")] for key in sd if key.endswith(".comfy_quant")]
    if not quantized_layers:
        return sd

    target_dtype = dtype or torch.float32
    for layer in quantized_layers:
        marker_key = f"{layer}.comfy_quant"
        weight_key = f"{layer}.weight"
        scale_key = f"{layer}.weight_scale"
        marker = sd.pop(marker_key)
        try:
            config = json.loads(marker.detach().cpu().numpy().tobytes().rstrip(b"\\x00").decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid ComfyUI quantization metadata for {layer}") from exc

        quant_format = config.get("format")
        if quant_format not in ("float8_e4m3fn", "float8_e5m2"):
            raise RuntimeError(
                f"Unsupported ComfyUI Qwen3-VL quantization format for {layer}: {quant_format}. "
                "Use a BF16 checkpoint or a float8_e4m3fn/e5m2 checkpoint."
            )
        if weight_key not in sd:
            raise RuntimeError(f"ComfyUI quantized Qwen3-VL layer is missing {weight_key}")
        if scale_key not in sd:
            raise RuntimeError(f"ComfyUI quantized Qwen3-VL layer is missing {scale_key}")

        weight = sd.pop(weight_key)
        scale = sd.pop(scale_key).to(device=weight.device, dtype=torch.float32)
        sd[weight_key] = (weight.float() * scale).to(dtype=target_dtype)

        # These are optional ComfyUI runtime-only parameters and are not part
        # of the standard Transformers Qwen3-VL state dict.
        sd.pop(f"{layer}.input_scale", None)

    logger.info("Dequantized %d ComfyUI scaled-FP8 Qwen3-VL layers to %s", len(quantized_layers), target_dtype)
    return sd


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL-4B from the vendored config and load weights from a local safetensors."""
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    # Keep source dtypes intact until ComfyUI scaled-FP8 metadata has been
    # interpreted.  Casting FP8 to BF16 before applying weight_scale would
    # preserve the encoded FP8 values, not recover the original weights.
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=None)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)
    sd = _dequantize_comfyui_fp8_state_dict(sd, dtype)

    info = model.load_state_dict(sd, strict=False, assign=True)
    # Qwen3-VL-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so the
    # checkpoint omits lm_head.weight; re-tie after loading to materialize it.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, max_length=max_length)
    processor = Qwen2TokenizerFast.from_pretrained(tokenizer_repo, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5
        self._layer_offloader: _Qwen3VLLayerOffloader | None = None

    def enable_layer_offload(self, device: torch.device | str, offload_percent: float = 1.0):
        """Use AIT-style GPU execution with CPU-resident Linear weights."""
        device = torch.device(device)
        if self._layer_offloader is not None:
            if self._layer_offloader.device == device:
                return self
            self.disable_layer_offload()

        self.qwen.to("cpu")
        self._layer_offloader = _Qwen3VLLayerOffloader(self.qwen, device, offload_percent)
        return self

    def disable_layer_offload(self):
        if self._layer_offloader is None:
            return
        self._layer_offloader.close()
        self._layer_offloader = None

    @property
    def device(self) -> torch.device:
        """Expose the wrapped Qwen device to the shared trainer interface."""
        if self._layer_offloader is not None:
            return self._layer_offloader.device
        return next(self.qwen.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Expose the wrapped Qwen dtype to the shared trainer interface."""
        return next(self.qwen.parameters()).dtype

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask
