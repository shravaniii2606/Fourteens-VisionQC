"""Reject unusable camera frames before running model inference."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config, db

logger = logging.getLogger(__name__)


def center_crop(frame: np.ndarray) -> np.ndarray:
    """Apply the model's resize-short-edge-to-256 then center-crop-to-224 geometry."""
    if frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    scale = 256.0 / min(height, width)
    resized_width = max(config.INPUT_SIZE, round(width * scale))
    resized_height = max(config.INPUT_SIZE, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top = (resized_height - config.INPUT_SIZE) // 2
    left = (resized_width - config.INPUT_SIZE) // 2
    return resized[top : top + config.INPUT_SIZE, left : left + config.INPUT_SIZE]


def calibrate_capture(good_images: list[np.ndarray]) -> dict[str, float]:
    """Store brightness, sharpness, and texture bounds from accepted examples."""
    brightness: list[float] = []
    sharpness: list[float] = []
    texture: list[float] = []
    dark_pixel_fractions: list[float] = []
    bright_pixel_fractions: list[float] = []
    for image in good_images:
        crop = center_crop(image)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        brightness.append(float(gray.mean()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        texture.append(float(gray.std()))
        dark_pixel_fractions.append(float(np.mean(gray <= 3)))
        bright_pixel_fractions.append(float(np.mean(gray >= 252)))
    result = {
        "brightness_mean": float(np.mean(brightness)),
        "brightness_min": max(0.0, min(brightness) - 3.0),
        "brightness_max": min(255.0, max(brightness) + 3.0),
        "sharpness_mean": float(np.mean(sharpness)),
        "sharpness_min": min(sharpness) * 0.5,
        "texture_min": max(4.0, min(texture) * 0.35),
        "dark_pixel_max": min(1.0, max(dark_pixel_fractions) + config.CAPTURE_PIXEL_FRACTION_MARGIN),
        "bright_pixel_max": min(1.0, max(bright_pixel_fractions) + config.CAPTURE_PIXEL_FRACTION_MARGIN),
    }
    db.set_setting("capture_calibration", result)
    return result


def check(frame: np.ndarray) -> dict[str, Any]:
    """Check a model-sized center crop and return metrics for UI diagnostics."""
    calibration = db.get_setting("capture_calibration", {})
    if frame is None or frame.size == 0:
        debug = {"brightness": None, "brightness_min": calibration.get("brightness_min"),
                 "brightness_max": calibration.get("brightness_max"), "sharpness": None,
                 "sharpness_min": calibration.get("sharpness_min"), "crop_size": None}
        logger.info("Capture check metrics: %s", debug)
        return {"ok": False, "reason": "No product in frame", "debug": debug}
    crop = center_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    mean = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    debug = {
        "brightness": mean,
        "brightness_min": calibration.get("brightness_min", config.CAPTURE_DARK_FALLBACK),
        "brightness_max": calibration.get("brightness_max", config.CAPTURE_BRIGHT_FALLBACK),
        "sharpness": sharpness,
        "sharpness_min": calibration.get("sharpness_min", config.BLUR_FALLBACK),
        "dark_pixel_fraction": float(np.mean(gray <= 3)),
        "dark_pixel_max": calibration.get("dark_pixel_max", config.DARK_PIXEL_LIMIT),
        "bright_pixel_fraction": float(np.mean(gray >= 252)),
        "bright_pixel_max": calibration.get("bright_pixel_max", config.BRIGHT_PIXEL_LIMIT),
        "texture": float(gray.std()),
        "texture_min": calibration.get("texture_min", 8.0),
        "crop_size": [int(gray.shape[1]), int(gray.shape[0])],
    }
    logger.info("Capture check metrics: %s", debug)
    if mean < calibration.get("brightness_min", config.CAPTURE_DARK_FALLBACK):
        return {"ok": False, "reason": "Too dark", "debug": debug}
    if mean > calibration.get("brightness_max", config.CAPTURE_BRIGHT_FALLBACK):
        return {"ok": False, "reason": "Too bright", "debug": debug}
    if float(np.mean(gray <= 3)) > calibration.get("dark_pixel_max", config.DARK_PIXEL_LIMIT):
        return {"ok": False, "reason": "Too dark", "debug": debug}
    if float(np.mean(gray >= 252)) > calibration.get("bright_pixel_max", config.BRIGHT_PIXEL_LIMIT):
        return {"ok": False, "reason": "Too bright", "debug": debug}
    if sharpness < calibration.get("sharpness_min", config.BLUR_FALLBACK):
        return {"ok": False, "reason": "Blurry, hold steady", "debug": debug}
    if float(gray.std()) < calibration.get("texture_min", 8.0):
        return {"ok": False, "reason": "No product in frame", "debug": debug}
    background_path = config.DATA_DIR / "empty_background.jpg"
    if background_path.exists():
        background = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)
        if background is not None:
            resized = cv2.resize(gray, (background.shape[1], background.shape[0]))
            if float(cv2.absdiff(resized, background).mean()) < config.EMPTY_DIFFERENCE_THRESHOLD:
                return {"ok": False, "reason": "No product in frame", "debug": debug}
    return {"ok": True, "reason": "", "debug": debug}
