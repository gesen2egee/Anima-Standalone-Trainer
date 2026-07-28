"""Keep or delete a generated image grid by running the MultiLabel TIMM tagger.

The process watches a dedicated capture directory, tags each image as soon as it
is written, and removes the complete directory if one image fails a condition.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from library.multilabel_timm import multilabel_timm_predict, unload_multilabel_timm_models


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def normalize_tag(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().split())


def score_for(tags: Mapping[str, float], wanted: str) -> float:
    normalized = normalize_tag(wanted)
    for tag, score in tags.items():
        if normalize_tag(tag) == normalized:
            return float(score)
    return 0.0


def evaluate_image(
    image_path: Path,
    repo_id: str,
    rating: str,
    rating_min_confidence: float,
    required_tag: str,
    tag_min_confidence: float,
) -> tuple[bool, dict[str, Any]]:
    rating_scores, general_scores, character_scores = multilabel_timm_predict(
        image_path,
        repo_id=repo_id,
        thresholds=0.0,
        use_tag_thresholds=False,
        fmt=("rating", "general", "character"),
    )
    result: dict[str, Any] = {"path": str(image_path), "rating": {}, "tag": {}}
    passed = True
    if rating:
        score = score_for(rating_scores, rating)
        result["rating"] = {"name": rating, "confidence": score}
        passed = passed and score >= rating_min_confidence
    if required_tag:
        score = score_for(general_scores, required_tag)
        if score == 0.0:
            score = score_for(character_scores, required_tag)
        result["tag"] = {"name": required_tag, "confidence": score}
        passed = passed and score >= tag_min_confidence
    result["passed"] = passed
    return passed, result


def iter_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=25)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--rating", default="")
    parser.add_argument("--rating-min-confidence", type=float, default=0.5)
    parser.add_argument("--required-tag", default="")
    parser.add_argument("--tag-min-confidence", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)

    directory = Path(args.image_dir)
    seen: set[Path] = set()
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        while len(seen) < args.expected_count:
            if not directory.exists():
                print(json.dumps({"type": "deleted", "reason": "capture directory removed"}), flush=True)
                return 1
            for image_path in iter_images(directory):
                if image_path in seen:
                    continue
                # Avoid attempting to read a PNG while PIL is still being written.
                try:
                    passed, result = evaluate_image(
                        image_path, args.repo_id, args.rating.strip(), args.rating_min_confidence,
                        args.required_tag.strip(), args.tag_min_confidence,
                    )
                except OSError:
                    continue
                seen.add(image_path)
                results.append(result)
                print(json.dumps({"type": "progress", "current": len(seen), "total": args.expected_count, **result}, ensure_ascii=False), flush=True)
                if not passed:
                    import shutil
                    shutil.rmtree(directory)
                    print(json.dumps({"type": "deleted", "reason": "filter mismatch", "results": results}, ensure_ascii=False), flush=True)
                    return 1
            if time.monotonic() - started > args.timeout_seconds:
                print(json.dumps({"type": "error", "error": "capture filter timed out"}), flush=True)
                return 2
            time.sleep(0.25)
        print(json.dumps({"type": "kept", "total": len(seen), "results": results}, ensure_ascii=False), flush=True)
        return 0
    finally:
        unload_multilabel_timm_models()


if __name__ == "__main__":
    raise SystemExit(main())
