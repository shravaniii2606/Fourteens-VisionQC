"""Evaluate one MVTec AD category using image-level PatchCore scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from backend import config
from backend.patchcore import PatchCore


def auroc(labels: list[int], scores: list[float]) -> float:
    """Calculate AUROC as the probability a positive outranks a negative."""
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        raise ValueError("Test data must include good and defective images")
    wins = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positive for neg in negative)
    return wins / (len(positive) * len(negative))


def main() -> None:
    """Fit on train/good and report image-level test metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=config.CATEGORY, help="MVTec AD category")
    parser.add_argument("--dataset", type=Path, default=config.DATA_DIR / "mvtec", help="MVTec AD root directory")
    args = parser.parse_args()

    category_dir = args.dataset / args.category
    good_paths = sorted((category_dir / "train" / "good").glob("*.png"))[:config.FIT_MAX_IMAGES]
    if len(good_paths) < 2:
        raise SystemExit(f"Need at least two images under {category_dir / 'train' / 'good'}")
    good_images = [Image.open(path).convert("RGB") for path in good_paths]
    model = PatchCore()
    model.category = args.category
    fit_stats = model.fit(good_images)
    calibration = model.calibrate(good_images)
    threshold = calibration["max"] + max(calibration["std"] * 0.05, 1e-6)

    labels: list[int] = []
    scores: list[float] = []
    for image_path in sorted((category_dir / "test").glob("*/*.png")):
        label = 0 if image_path.parent.name == "good" else 1
        with Image.open(image_path) as image:
            result = model.score(image.convert("RGB"))
        labels.append(label)
        scores.append(result["max_score"])
    detected = sum(label == 1 and score > threshold for label, score in zip(labels, scores))
    defective_count = sum(labels)
    false_alarms = sum(label == 0 and score > threshold for label, score in zip(labels, scores))
    good_count = len(labels) - defective_count
    print(f"Category: {args.category}")
    print(f"Fit: {fit_stats['images']} good images, {fit_stats['patches']} memory patches")
    print(f"Calibration: {calibration}")
    print(f"Chosen threshold: {threshold:.6f}")
    print(f"AUROC: {auroc(labels, scores):.4f}")
    print(f"Detection rate: {detected / defective_count:.4f}")
    print(f"False alarm rate: {false_alarms / good_count:.4f}")


if __name__ == "__main__":
    main()
