"""Ordered capture, alignment, anomaly, and supervisor-threshold decision."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from . import align as alignment
from . import capture_check, config, db
from .patchcore import PatchCore


def _recapture(reason: str, debug: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a consistent recapture result without a heatmap."""
    return {
        "verdict": "RECAPTURE",
        "reason": reason,
        "score": 0.0,
        "confidence_pct": 0.0,
        "hotspot": None,
        "heatmap_png_base64": None,
        "debug": debug or {},
    }


def _inspection_debug(capture: dict[str, Any], aligned: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine capture metrics with alignment shift metrics for persistence."""
    debug = dict(capture.get("debug") or {})
    if aligned is not None:
        debug["alignment_shift"] = aligned.get("shift")
        debug["alignment_rotation"] = aligned.get("rotation")
    return debug


def decide(frame: np.ndarray, model: PatchCore) -> dict[str, Any]:
    """Run all inspection gates in order and return the chosen verdict."""
    capture = capture_check.check(frame)
    if not capture["ok"]:
        return _recapture(capture["reason"], capture.get("debug"))
    aligned = alignment.align(frame)
    if not aligned["ok"]:
        return _recapture(aligned["reason"], _inspection_debug(capture, aligned))
    working_frame = aligned["aligned_image"]
    try:
        result = model.score(working_frame)
    except RuntimeError as error:
        if "Fit the model first" in str(error):
            raise
        raise
    anomaly_map = result["anomaly_map"]
    score = float(result["max_score"])
    threshold = db.get_threshold()
    low_threshold = max(config.SPREAD_THRESHOLD * max(threshold, 1e-6), model.calibration.get("mean", 0.0))
    spread = float(np.mean(anomaly_map > low_threshold))
    if spread > config.SPREAD_FRACTION:
        return _recapture("Whole image looks off, check lighting or position", _inspection_debug(capture, aligned))
    verdict = "FAIL" if score > threshold else "PASS"
    reason = "Anomaly score exceeds threshold" if verdict == "FAIL" else "Within threshold"
    scaled = np.clip(anomaly_map / max(threshold * 2.0, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(working_frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.5, colored_bgr, 0.5, 0)
    encoded_ok, encoded = cv2.imencode(".png", overlay)
    if not encoded_ok:
        raise ValueError("Could not encode the heatmap")
    hotspot = result["hotspot"]
    return {
        "verdict": verdict,
        "reason": reason,
        "score": score,
        "confidence_pct": model.confidence_pct(score, threshold),
        "hotspot": {"x": hotspot[0], "y": hotspot[1]},
        "heatmap_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "aligned_frame": working_frame,
        "debug": _inspection_debug(capture, aligned),
    }
