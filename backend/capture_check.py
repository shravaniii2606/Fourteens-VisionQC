"""Reject unusable camera frames before running model inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import config, db


def calibrate_capture(good_images: list[np.ndarray]) -> dict[str, float]:
    """Store brightness, sharpness, and texture bounds from accepted examples."""
    brightness: list[float] = []
    sharpness: list[float] = []
    texture: list[float] = []
    for image in good_images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        brightness.append(float(gray.mean()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        texture.append(float(gray.std()))
    result = {
        "brightness_min": max(0.0, min(brightness) - 3.0),
        "brightness_max": min(255.0, max(brightness) + 3.0),
        "sharpness_min": max(config.BLUR_FALLBACK, min(sharpness) * 0.5),
        "texture_min": max(4.0, min(texture) * 0.35),
    }
    db.set_setting("capture_calibration", result)
    return result


def check(frame: np.ndarray) -> dict[str, Any]:
    """Check brightness, blur, and likely product presence on an RGB frame."""
    if frame is None or frame.size == 0:
        return {"ok": False, "reason": "No product in frame"}
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    calibration = db.get_setting("capture_calibration", {})
    mean = float(gray.mean())
    if mean < calibration.get("brightness_min", config.CAPTURE_DARK_FALLBACK):
        return {"ok": False, "reason": "Too dark"}
    if mean > calibration.get("brightness_max", config.CAPTURE_BRIGHT_FALLBACK):
        return {"ok": False, "reason": "Too bright"}
    if float(np.mean(gray <= 3)) > config.DARK_PIXEL_LIMIT:
        return {"ok": False, "reason": "Too dark"}
    if float(np.mean(gray >= 252)) > config.BRIGHT_PIXEL_LIMIT:
        return {"ok": False, "reason": "Too bright"}
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < calibration.get("sharpness_min", config.BLUR_FALLBACK):
        return {"ok": False, "reason": "Blurry, hold steady"}
    if float(gray.std()) < calibration.get("texture_min", 8.0):
        return {"ok": False, "reason": "No product in frame"}
    background_path = config.DATA_DIR / "empty_background.jpg"
    if background_path.exists():
        background = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)
        if background is not None:
            resized = cv2.resize(gray, (background.shape[1], background.shape[0]))
            if float(cv2.absdiff(resized, background).mean()) < config.EMPTY_DIFFERENCE_THRESHOLD:
                return {"ok": False, "reason": "No product in frame"}
    return {"ok": True, "reason": ""}
