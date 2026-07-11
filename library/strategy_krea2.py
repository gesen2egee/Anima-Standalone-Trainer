"""Dataset strategies for Krea 2 on top of the existing Anima pipeline."""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from library.krea2 import krea2_sampling, krea2_utils
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.strategy_base import TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy


class Krea2TokenizeStrategy(TokenizeStrategy):
    """Provide ordinary token tensors required by the existing dataset.

    Krea 2 training consumes the dedicated cached conditioner output below. These
    token tensors are retained so the shared dataset/collator can keep its normal
    interface and so uncached usage fails with a useful message in the trainer.
    """

    def __init__(self, tokenizer_repo: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)
        self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        texts = [text] if isinstance(text, str) else text
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        return [encoded["input_ids"], encoded["attention_mask"]]


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    """Krea 2 requires pre-cached Qwen3-VL outputs for training."""

    def encode_tokens(self, tokenize_strategy, models, tokens):
        raise RuntimeError(
            "Krea 2 training requires --cache_text_encoder_outputs. "
            "Use the Krea2 text-cache step before training."
        )


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Cache fixed-width, valid-prefix Qwen3-VL hidden states.

    The conditioner removes interior padding before this strategy pads each item
    to the configured width. This preserves the Krea2 attention semantics while
    remaining compatible with the existing dataset collator's tensor stacking.
    """

    KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_krea2_te.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, max_length: int = 512):
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.max_length = max_length

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            data = np.load(npz_path)
            return "prompt_embeds" in data and "attn_mask" in data
        except Exception:
            return False

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        return [data["prompt_embeds"], data["attn_mask"]]

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, infos: List):
        if not models or models[0] is None:
            raise ValueError("Krea2 text encoder is required to build the text cache")

        encoder = models[0]
        prompts = [info.caption for info in infos]
        with torch.no_grad():
            hidden, mask = krea2_utils.get_krea2_prompt_embeds(encoder, prompts)
            hidden, mask = krea2_sampling.gather_valid_text(hidden, mask.bool())

        if hidden.shape[1] > self.max_length:
            raise ValueError(
                f"Krea2 text cache length {hidden.shape[1]} exceeds --krea2_max_token_length={self.max_length}"
            )
        padded_hidden = hidden.new_zeros(hidden.shape[0], self.max_length, hidden.shape[2], hidden.shape[3])
        padded_mask = torch.zeros(hidden.shape[0], self.max_length, dtype=torch.bool, device=mask.device)
        padded_hidden[:, : hidden.shape[1]] = hidden
        padded_mask[:, : mask.shape[1]] = mask

        hidden_np = padded_hidden.float().cpu().numpy()
        mask_np = padded_mask.cpu().numpy().astype(np.uint8)
        for index, info in enumerate(infos):
            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, prompt_embeds=hidden_np[index], attn_mask=mask_np[index])
            else:
                info.text_encoder_outputs = (hidden_np[index], mask_np[index])


class Krea2LatentsCachingStrategy(AnimaLatentsCachingStrategy):
    """Qwen-Image VAE cache with a Krea2-specific filename suffix."""

    KREA2_LATENTS_NPZ_SUFFIX = "_krea2.npz"

    @property
    def cache_suffix(self) -> str:
        return self.KREA2_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.KREA2_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(self, npz_path: str, bucket_reso):
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)
