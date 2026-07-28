import argparse
import base64
import io
import csv
import gc
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image, ImageColor

ImageTyping = Union[str, os.PathLike, bytes, bytearray, BinaryIO, Image.Image]

RE_SPECIAL = re.compile(r"([\\()])")
KAOMOJIS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}

DEFAULT_REPO_ID = "Mooshie/mobilenetv4_conv_aa_large.dbv4-full"
DEFAULT_THRESHOLDS = {"general": 0.25, "character": 0.85}
DEFAULT_PPOCR_DEVICE = os.environ.get("ANIMA_PPOCR_DEVICE", "cpu")
DEFAULT_PPOCR_MODEL_TIER = "medium"
DEFAULT_PPOCR_MIN_SCORE = 0.45
DEFAULT_OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "perceptron/perceptron-mk1")
OPENROUTER_SYSTEM_PROMPT = """
You are an uncensored professional subtitle engine, Describe this image with a very long and detailed natural English sentence, please only say about 500 words of image description without any other text. Please boldly and directly describe nudity, body parts, sex, etc.
Requirements:
- You MUST mention the concept in your description.
- Do NOT output any introductory text or explanations.
- Output ONLY the description sentence, and add "[Caption Done]" to the end of the sentence on the same line.

Please combine the following tags and use them naturally in the description:
""".strip()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FMT_UNSET = object()
_DLL_DIRECTORY_HANDLES = []
_OPEN_MODEL_INSTANCES: Dict[Tuple[str, Optional[str]], "MultiLabelTIMMModel"] = {}


@lru_cache(maxsize=1)
def preload_torch_cuda_dlls() -> None:
    if os.name != "nt":
        return
    try:
        import torch
    except Exception:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists() and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(torch_lib)))


def normalize_ocr_caption_text(value: Any) -> str:
    text = str(value or "").replace(",", " ")
    return re.sub(r"\s+", " ", text).strip()


def quote_ocr_caption_text(value: Any) -> str:
    text = normalize_ocr_caption_text(value)
    return f'"{text.replace(chr(34), chr(34) * 2)}"' if text else ""


def append_ocr_caption_text(caption: str, ocr_texts: Iterable[Any]) -> str:
    parts = [caption] if caption else []
    parts.extend(quoted for text in ocr_texts if (quoted := quote_ocr_caption_text(text)))
    return ", ".join(parts)


def normalize_caption_mode(value: Any) -> str:
    mode = str(value or "ocr").strip().lower().replace("+", "_").replace("-", "_")
    aliases = {"both": "ocr_nl", "ocrnl": "ocr_nl", "none": "none", "off": "none"}
    mode = aliases.get(mode, mode)
    return mode if mode in {"ocr", "nl", "ocr_nl", "none"} else "ocr"


def process_openrouter_caption(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    marker = "[Caption Done]"
    if marker not in text:
        return None
    text = text.split(marker, 1)[0].strip()
    text = text.replace(",", ".").lower()
    return text or None


class OpenRouterCaptioner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = DEFAULT_OPENROUTER_API_URL,
        model: str = DEFAULT_OPENROUTER_MODEL,
    ):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.api_url = (api_url or DEFAULT_OPENROUTER_API_URL).strip()
        self.model = (model or DEFAULT_OPENROUTER_MODEL).strip()
        self.client = None
        if self.api_key:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)

    @staticmethod
    def _image_data_url(image_path: Union[str, os.PathLike]) -> str:
        with Image.open(image_path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def caption(self, image_path: Union[str, os.PathLike], concept: str, tags: str) -> Optional[str]:
        if self.client is None:
            return None
        prompt = f"Concept: {concept.strip()}\nTags: {tags.strip()}"
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}},
        ]
        for _attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=1000,
                )
                caption = process_openrouter_caption(response.choices[0].message.content or "")
                if caption is not None:
                    return caption
            except Exception:
                continue
        return None


def extract_paddleocr_texts(results: Iterable[Any], min_score: float = DEFAULT_PPOCR_MIN_SCORE) -> list[str]:
    texts = []
    for result in results:
        payload = getattr(result, "json", result)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, Mapping):
            continue
        payload = payload.get("res", payload)
        if not isinstance(payload, Mapping):
            continue
        rec_texts = payload.get("rec_texts", [])
        rec_scores = payload.get("rec_scores", [])
        for index, text in enumerate(rec_texts):
            score = float(rec_scores[index]) if index < len(rec_scores) else 1.0
            normalized = normalize_ocr_caption_text(text)
            if normalized and score >= min_score:
                texts.append(normalized)
    return texts


class PaddleOcrCaptioner:
    def __init__(
        self,
        device: str = DEFAULT_PPOCR_DEVICE,
        model_tier: str = DEFAULT_PPOCR_MODEL_TIER,
        min_score: float = DEFAULT_PPOCR_MIN_SCORE,
    ):
        tier = model_tier.strip().lower()
        if tier not in {"medium", "small"}:
            raise ValueError("PP-OCRv6 model tier must be medium or small")

        if device.lower().startswith("gpu"):
            preload_torch_cuda_dlls()
            import torch  # noqa: F401 - preload CUDA/cuDNN DLLs before ONNX Runtime
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")

        from paddleocr import PaddleOCR

        self.min_score = float(min_score)
        self.pipeline = PaddleOCR(
            text_detection_model_name=f"PP-OCRv6_{tier}_det",
            text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
            engine="onnxruntime",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict_texts(self, image_path: Union[str, os.PathLike]) -> list[str]:
        return extract_paddleocr_texts(self.pipeline.predict(str(image_path)), min_score=self.min_score)


def remove_underline(tag: str) -> str:
    tag = tag.strip()
    return tag if tag in KAOMOJIS else tag.replace("_", " ")


def tags_to_text(
    tags: Mapping[str, float],
    use_spaces: bool = False,
    use_escape: bool = True,
    include_score: bool = False,
    score_descend: bool = True,
) -> str:
    text_items = []
    tags_pairs = tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        text = remove_underline(tag) if use_spaces else tag
        if use_escape:
            text = re.sub(RE_SPECIAL, r"\\\1", text)
        if include_score:
            text = f"({text}:{score:.3f})"
        text_items.append(text)
    return ", ".join(text_items)


def split_hf_repo_id(repo_id: str) -> Tuple[str, Optional[str]]:
    parts = repo_id.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return repo_id, None


def _hf_download(repo_id: str, filename: str, token: Optional[str] = None) -> str:
    base_repo_id, subfolder = split_hf_repo_id(repo_id)
    return hf_hub_download(
        repo_id=base_repo_id,
        repo_type="model",
        filename=filename,
        subfolder=subfolder,
        token=token,
    )


def _has_alpha_channel(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    if getattr(image, "palette", None):
        try:
            image.palette.getcolor((0, 0, 0, 0))
            return True
        except ValueError:
            pass
    return "transparency" in image.info


def load_image(image: ImageTyping, mode: Optional[str] = None, force_background: Optional[str] = "white") -> Image.Image:
    if isinstance(image, (str, os.PathLike, bytes, bytearray)) or hasattr(image, "read"):
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        pass
    else:
        raise TypeError(f"Unknown image type - {image!r}.")

    if _has_alpha_channel(image) and force_background is not None:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, force_background)
        background.paste(image, (0, 0), mask=image.getchannel("A"))
        image = background.convert("RGB")
    if mode is not None and image.mode != mode:
        image = image.convert(mode)
    return image


class PillowCompose:
    def __init__(self, transforms: Sequence[Any]):
        self.transforms = transforms

    def __call__(self, image: Any) -> Any:
        value = image
        for transform in self.transforms:
            value = transform(value)
        return value


class PillowResize:
    RESAMPLE = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "box": Image.BOX,
        "hamming": Image.HAMMING,
        "lanczos": Image.LANCZOS,
    }

    def __init__(self, size: Union[int, Sequence[int]], interpolation: str = "bilinear", max_size: Optional[int] = None,
                 antialias: bool = True):
        self.size = size
        self.interpolation = self.RESAMPLE[str(interpolation).lower()]
        self.max_size = max_size
        self.antialias = antialias

    def _target_size(self, image: Image.Image) -> Tuple[int, int]:
        width, height = image.size
        if isinstance(self.size, int) or (isinstance(self.size, (list, tuple)) and len(self.size) == 1):
            size = self.size if isinstance(self.size, int) else self.size[0]
            if width < height:
                out_w = size
                out_h = int(size * height / width)
            else:
                out_h = size
                out_w = int(size * width / height)
            if self.max_size is not None and max(out_w, out_h) > self.max_size:
                if out_h > out_w:
                    out_w = int(self.max_size * out_w / out_h)
                    out_h = self.max_size
                else:
                    out_h = int(self.max_size * out_h / out_w)
                    out_w = self.max_size
            return out_w, out_h
        return int(self.size[1]), int(self.size[0])

    def __call__(self, image: Image.Image) -> Image.Image:
        target = self._target_size(image)
        if target == image.size:
            return image
        if self.interpolation in {Image.BILINEAR, Image.BICUBIC}:
            return image.resize(target, self.interpolation, reducing_gap=None if self.antialias else 1.0)
        return image.resize(target, self.interpolation)


class PillowCenterCrop:
    def __init__(self, size: Union[int, Sequence[int]]):
        if isinstance(size, int):
            self.size = (size, size)
        elif len(size) == 1:
            self.size = (int(size[0]), int(size[0]))
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, image: Image.Image) -> Image.Image:
        crop_h, crop_w = self.size
        width, height = image.size
        if width < crop_w or height < crop_h:
            padded = Image.new(image.mode, (max(width, crop_w), max(height, crop_h)), (0, 0, 0))
            padded.paste(image, ((padded.width - width) // 2, (padded.height - height) // 2))
            image = padded
            width, height = image.size
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))


def _parse_size(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise TypeError("Size must be int or tuple/list of two ints")


def _parse_color_to_rgba(color: Union[str, int, Sequence[int]]) -> Tuple[int, int, int, int]:
    if isinstance(color, str):
        rgba = ImageColor.getrgb(color)
        return tuple([*rgba, *((255,) * (4 - len(rgba)))])
    if isinstance(color, int):
        return color, color, color, 255
    if isinstance(color, (list, tuple)):
        return tuple([*color, *((255,) * (4 - len(color)))])
    raise TypeError(f"Invalid color type: {type(color)}")


def _parse_color_to_mode(color: Union[str, int, Sequence[int]], mode: str) -> Any:
    rgba = _parse_color_to_rgba(color)
    if mode == "L":
        return int(0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
    if mode == "LA":
        gray = int(0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
        return gray, rgba[3]
    if mode == "RGB":
        return rgba[:3]
    if mode == "RGBA":
        return rgba
    return rgba[:3] if len(mode) >= 3 else rgba[0]


class PillowPadToSize:
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        background_color: Union[str, int, Sequence[int]] = "white",
        interpolation: str = "bilinear",
    ):
        self.size = _parse_size(size)
        self.background_color = background_color
        self.interpolation = PillowResize.RESAMPLE[str(interpolation).lower()]

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"pic should be PIL Image. Got {type(image)}")
        target_w, target_h = self.size
        original_w, original_h = image.size
        ratio = min(target_w / original_w, target_h / original_h)
        new_w = round(original_w * ratio)
        new_h = round(original_h * ratio)
        resized = image.resize((new_w, new_h), self.interpolation)
        canvas = Image.new(image.mode, (target_w, target_h), _parse_color_to_mode(self.background_color, image.mode))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas


class PillowToTensor:
    def __call__(self, image: Image.Image) -> np.ndarray:
        if image.mode == "I":
            return np.array(image, np.int32, copy=True)[None, ...]
        if image.mode == "I;16":
            return np.array(image, np.int16, copy=True)[None, ...]
        if image.mode == "F":
            return np.array(image, np.float32, copy=True)[None, ...]
        array = np.array(image, copy=True)
        if image.mode == "L":
            return array.reshape((1,) + array.shape).astype(np.float32) / 255
        if image.mode in ("RGB", "RGBA", "YCbCr", "CMYK"):
            return array.transpose((2, 0, 1)).astype(np.float32) / 255
        return array.astype(np.float32)[None, ...] / 255


class PillowMaybeToTensor:
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        return image if isinstance(image, np.ndarray) else PillowToTensor()(image)


class PillowNormalize:
    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]], inplace: bool = False):
        self.mean = np.array(mean if isinstance(mean, (list, tuple)) else [mean], dtype=np.float32)
        self.std = np.array(std if isinstance(std, (list, tuple)) else [std], dtype=np.float32)
        self.inplace = inplace

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if not self.inplace:
            array = array.copy()
        array -= self.mean.reshape(-1, 1, 1)
        array /= self.std.reshape(-1, 1, 1)
        return array


def create_pillow_transforms(config: Union[list, dict]) -> Any:
    creators = {
        "resize": lambda item: PillowResize(**item),
        "center_crop": lambda item: PillowCenterCrop(**item),
        "pad_to_size": lambda item: PillowPadToSize(**item),
        "to_tensor": lambda item: PillowToTensor(),
        "maybe_to_tensor": lambda item: PillowMaybeToTensor(),
        "normalize": lambda item: PillowNormalize(**item),
    }
    if isinstance(config, list):
        return PillowCompose([create_pillow_transforms(item) for item in config])
    if isinstance(config, dict):
        item = dict(config)
        transform_type = item.pop("type")
        if transform_type not in creators:
            raise ValueError(f"Unsupported transform type: {transform_type}")
        return creators[transform_type](item)
    raise TypeError(f"Unknown transform config: {config!r}")


def get_onnx_provider(provider: Optional[str] = None) -> str:
    preload_torch_cuda_dlls()
    import onnxruntime as ort

    aliases = {"gpu": "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider"}
    if provider:
        provider = aliases.get(provider.lower(), provider)
        for available in ort.get_available_providers():
            if provider.lower() == available.lower() or f"{provider}ExecutionProvider".lower() == available.lower():
                return available
        raise ValueError(f"Unsupported ONNX provider {provider!r}; available: {ort.get_available_providers()!r}")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def open_onnx_model(path: str, provider: Optional[str] = None):
    preload_torch_cuda_dlls()
    import onnxruntime as ort

    selected = get_onnx_provider(provider or os.environ.get("ONNX_MODE"))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [selected]
    if selected != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(path, options, providers=providers)


def vreplace(value: Any, mapping: Mapping[Any, Any]) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)([vreplace(item, mapping) for item in value])
    if isinstance(value, dict):
        return type(value)({key: vreplace(item, mapping) for key, item in value.items()})
    try:
        hash(value)
    except TypeError:
        return value
    return mapping.get(value, value)


class MultiLabelTIMMModel:
    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self._model = None
        self._tags = None
        self._preprocess = None
        self._category_names: Dict[Any, str] = {}
        self._default_category_thresholds = None

    def _get_hf_token(self) -> Optional[str]:
        return self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def unload(self) -> None:
        self._model = None
        self._tags = None
        self._preprocess = None
        self._category_names = {}
        self._default_category_thresholds = None

    def _open_model(self):
        if self._model is None:
            self._model = open_onnx_model(_hf_download(self.repo_id, "model.onnx", self._get_hf_token()))
        return self._model

    def _open_tags(self):
        if self._tags is None:
            tags_path = _hf_download(self.repo_id, "selected_tags.csv", self._get_hf_token())
            with open(tags_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            categories_path = _hf_download(self.repo_id, "categories.json", self._get_hf_token())
            with open(categories_path, "r", encoding="utf-8") as f:
                category_names = {item["category"]: item["name"] for item in json.load(f)}
            for row in rows:
                category = _coerce_category(row["category"])
                row["category"] = category
                if "best_threshold" in row and row["best_threshold"] != "":
                    row["best_threshold"] = float(row["best_threshold"])
                self._category_names[category] = category_names.get(category, category_names.get(str(category), str(category)))
            self._tags = rows
        return self._tags

    def _open_preprocess(self):
        if self._preprocess is None:
            with open(_hf_download(self.repo_id, "preprocess.json", self._get_hf_token()), "r", encoding="utf-8") as f:
                data = json.load(f)
            self._preprocess = create_pillow_transforms(data["val"]), create_pillow_transforms(data["test"])
        return self._preprocess

    def _open_default_category_thresholds(self):
        if self._default_category_thresholds is None:
            self._default_category_thresholds = {}
            try:
                threshold_path = _hf_download(self.repo_id, "thresholds.csv", self._get_hf_token())
            except EntryNotFoundError:
                return self._default_category_thresholds
            with open(threshold_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._default_category_thresholds[_coerce_category(row["category"])] = float(row["threshold"])
        return self._default_category_thresholds

    def _raw_predict(self, image: ImageTyping, preprocessor: str = "test") -> Dict[str, np.ndarray]:
        image = load_image(image, force_background="white", mode="RGB")
        model = self._open_model()
        val_trans, test_trans = self._open_preprocess()
        if preprocessor == "test":
            transform = test_trans
        elif preprocessor == "val":
            transform = val_trans
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor!r}; expected 'test' or 'val'.")
        input_array = transform(image)[None, ...]
        input_name = model.get_inputs()[0].name
        output_names = [output.name for output in model.get_outputs()]
        output_values = model.run(output_names, {input_name: input_array})
        return {name: value[0] for name, value in zip(output_names, output_values)}

    def predict(
        self,
        image: ImageTyping,
        preprocessor: str = "test",
        thresholds: Optional[Union[float, Mapping[Any, float]]] = None,
        use_tag_thresholds: bool = True,
        fmt: Any = FMT_UNSET,
    ) -> Any:
        rows = self._open_tags()
        values = self._raw_predict(image, preprocessor=preprocessor)
        prediction = values["prediction"]
        if fmt is FMT_UNSET:
            fmt = tuple(self._category_names[category] for category in sorted({row["category"] for row in rows}))
        tag_values = {}
        default_thresholds = self._open_default_category_thresholds()
        for category in sorted({row["category"] for row in rows}):
            category_name = self._category_names[category]
            indexed_rows = [(index, row) for index, row in enumerate(rows) if row["category"] == category]
            threshold = _resolve_threshold(category, category_name, indexed_rows, thresholds, use_tag_thresholds, default_thresholds)
            category_tags = {}
            for index, row in indexed_rows:
                score = float(prediction[index])
                row_threshold = threshold[index] if isinstance(threshold, dict) else threshold
                if score >= row_threshold:
                    category_tags[row["name"]] = score
            values[category_name] = dict(sorted(category_tags.items(), key=lambda x: (-x[1], x[0])))
            tag_values.update(values[category_name])
        values["tag"] = tag_values
        return vreplace(fmt, values)


def _coerce_category(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _resolve_threshold(category, category_name, indexed_rows, thresholds, use_tag_thresholds, default_thresholds):
    if isinstance(thresholds, (int, float)):
        return float(thresholds)
    if isinstance(thresholds, Mapping):
        if category in thresholds:
            return float(thresholds[category])
        if category_name in thresholds:
            return float(thresholds[category_name])
    if use_tag_thresholds and any("best_threshold" in row for _, row in indexed_rows):
        return {
            index: float(row.get("best_threshold") or 0.4)
            for index, row in indexed_rows
        }
    return float(default_thresholds.get(category, 0.4))


@lru_cache(maxsize=4)
def _open_model_for_repo(repo_id: str, hf_token: Optional[str] = None) -> MultiLabelTIMMModel:
    model = MultiLabelTIMMModel(repo_id=repo_id, hf_token=hf_token)
    _OPEN_MODEL_INSTANCES[(repo_id, hf_token)] = model
    return model


def unload_multilabel_timm_models() -> None:
    for model in list(_OPEN_MODEL_INSTANCES.values()):
        model.unload()
    _OPEN_MODEL_INSTANCES.clear()
    _open_model_for_repo.cache_clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def multilabel_timm_predict(
    image: ImageTyping,
    repo_id: str,
    preprocessor: str = "test",
    thresholds: Optional[Union[float, Mapping[Any, float]]] = None,
    use_tag_thresholds: bool = True,
    fmt: Any = FMT_UNSET,
    hf_token: Optional[str] = None,
) -> Any:
    return _open_model_for_repo(repo_id, hf_token).predict(
        image=image,
        preprocessor=preprocessor,
        thresholds=thresholds,
        use_tag_thresholds=use_tag_thresholds,
        fmt=fmt,
    )


multilabel_timm_predict_patched = multilabel_timm_predict


def compose_caption_text(
    rating: Mapping[str, float],
    general: Mapping[str, float],
    character: Mapping[str, float],
    include_char: bool = True,
    include_rating: bool = True,
    include_general: bool = True,
) -> str:
    parts = []
    if include_char and character:
        parts.append(tags_to_text(character, use_spaces=True, use_escape=False))
    if include_rating:
        rating_value = max(rating, key=rating.get) if rating else "general"
        parts.append(remove_underline(rating_value))
    if include_general and general:
        parts.append(tags_to_text(general, use_spaces=True, use_escape=False))
    return ", ".join(part for part in parts if part)


def iter_image_paths(image_dir: Union[str, os.PathLike], image_list: Optional[Union[str, os.PathLike]] = None) -> Iterable[Path]:
    if image_list:
        for line in Path(image_list).read_text(encoding="utf-8").splitlines():
            path = Path(line.strip())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
        return

    root = Path(image_dir)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_captions_for_directory(
    image_dir: Union[str, os.PathLike],
    caption_extension: str = ".txt",
    repo_id: str = DEFAULT_REPO_ID,
    thresholds: Optional[Mapping[str, float]] = None,
    include_char: bool = True,
    include_rating: bool = True,
    include_general: bool = True,
    include_ocr: bool = True,
    ocr_device: str = DEFAULT_PPOCR_DEVICE,
    ocr_model_tier: str = DEFAULT_PPOCR_MODEL_TIER,
    ocr_min_score: float = DEFAULT_PPOCR_MIN_SCORE,
    caption_mode: str = "ocr",
    concept: str = "",
    concept_map: Optional[Mapping[str, str]] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_api_url: str = DEFAULT_OPENROUTER_API_URL,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    image_list: Optional[Union[str, os.PathLike]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, int]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    image_paths = list(iter_image_paths(image_dir, image_list))
    written = 0
    failed = 0
    mode = normalize_caption_mode(caption_mode)
    if not include_ocr:
        mode = {"ocr": "none", "ocr_nl": "nl"}.get(mode, mode)
    if progress_callback:
        progress_callback({"type": "start", "total": len(image_paths), "written": written, "failed": failed})
    ocr_captioner = (
        PaddleOcrCaptioner(device=ocr_device, model_tier=ocr_model_tier, min_score=ocr_min_score)
        if mode in {"ocr", "ocr_nl"} and image_paths
        else None
    )
    nl_captioner = (
        OpenRouterCaptioner(
            api_key=openrouter_api_key,
            api_url=openrouter_api_url,
            model=openrouter_model,
        )
        if mode in {"nl", "ocr_nl"} and image_paths
        else None
    )
    for index, image_path in enumerate(image_paths, start=1):
        error = None
        try:
            rating, general, character = multilabel_timm_predict(
                image_path,
                repo_id=repo_id,
                thresholds=thresholds,
                fmt=("rating", "general", "character"),
            )
            caption = compose_caption_text(
                rating=rating,
                general=general,
                character=character,
                include_char=include_char,
                include_rating=include_rating,
                include_general=include_general,
            )
            if ocr_captioner is not None:
                caption = append_ocr_caption_text(caption, ocr_captioner.predict_texts(image_path))
            if nl_captioner is not None:
                tag_caption = compose_caption_text(
                    rating=rating,
                    general=general,
                    character=character,
                    include_char=include_char,
                    include_rating=include_rating,
                    include_general=include_general,
                )
                nl_caption = nl_captioner.caption(
                    image_path,
                    concept=(concept_map or {}).get(str(image_path), concept),
                    tags=tag_caption,
                )
                if nl_caption:
                    caption = f"{caption}\n{nl_caption}"
                else:
                    # NL 失敗重試後仍沒有完成標記時，依需求只保留原始 Tags。
                    caption = tag_caption
            image_path.with_suffix(caption_extension).write_text(caption, encoding="utf-8")
            written += 1
        except Exception as exc:
            failed += 1
            error = str(exc)
        if progress_callback:
            progress_callback({
                "type": "progress",
                "current": index,
                "total": len(image_paths),
                "path": str(image_path),
                "written": written,
                "failed": failed,
                "error": error,
            })
    return {"total": len(image_paths), "written": written, "failed": failed}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--caption-extension", default=".txt")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--general-threshold", type=float, default=0.25)
    parser.add_argument("--character-threshold", type=float, default=0.85)
    parser.add_argument("--include-char", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-rating", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-general", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-ocr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ocr-device", default=DEFAULT_PPOCR_DEVICE)
    parser.add_argument("--ocr-model-tier", choices=("medium", "small"), default=DEFAULT_PPOCR_MODEL_TIER)
    parser.add_argument("--ocr-min-score", type=float, default=DEFAULT_PPOCR_MIN_SCORE)
    parser.add_argument("--caption-mode", choices=("ocr", "nl", "ocr_nl", "none"), default="ocr")
    parser.add_argument("--concept", default="")
    parser.add_argument("--concept-map", default=None)
    parser.add_argument("--openrouter-api-url", default=DEFAULT_OPENROUTER_API_URL)
    parser.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--image-list", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    concept_map = {}
    if args.concept_map:
        concept_map = json.loads(Path(args.concept_map).read_text(encoding="utf-8"))

    def emit(event: Dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        result = write_captions_for_directory(
            image_dir=args.image_dir,
            caption_extension=args.caption_extension,
            repo_id=args.repo_id,
            thresholds={"general": args.general_threshold, "character": args.character_threshold},
            include_char=args.include_char,
            include_rating=args.include_rating,
            include_general=args.include_general,
            include_ocr=args.include_ocr,
            ocr_device=args.ocr_device,
            ocr_model_tier=args.ocr_model_tier,
            ocr_min_score=args.ocr_min_score,
            caption_mode=args.caption_mode,
            concept=args.concept,
            concept_map=concept_map,
            openrouter_api_url=args.openrouter_api_url,
            openrouter_model=args.openrouter_model,
            image_list=args.image_list,
            progress_callback=emit,
        )
        emit({"type": "done", **result})
        return 0 if result["failed"] == 0 else 1
    finally:
        unload_multilabel_timm_models()


if __name__ == "__main__":
    raise SystemExit(main())
