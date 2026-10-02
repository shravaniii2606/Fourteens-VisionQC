"""ORB/RANSAC image alignment against the configured product reference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import config


def align(frame: np.ndarray) -> dict[str, Any]:
    """Align an RGB frame to reference.jpg or report a position mismatch."""
    if not config.ALIGN_ENABLED:
        return {"ok": True, "aligned_image": frame, "inliers": 0, "shift": 0.0, "rotation": 0.0, "reason": ""}
    reference_path = config.DATA_DIR / "reference.jpg"
    reference_bgr = cv2.imread(str(reference_path))
    if reference_bgr is None:
        return {"ok": True, "aligned_image": frame, "inliers": 0, "shift": None, "rotation": None, "reason": ""}
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=config.ORB_MAX_FEATURES)
    reference_points, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    frame_points, frame_descriptors = detector.detectAndCompute(frame_gray, None)
    if reference_descriptors is None or frame_descriptors is None:
        return {"ok": False, "aligned_image": None, "inliers": 0, "shift": None, "rotation": None, "reason": "Product shifted, move into position"}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < config.ORB_RATIO_TEST * second.distance]
    if len(matches) < config.ALIGN_MIN_INLIERS:
        return {"ok": False, "aligned_image": None, "inliers": 0, "shift": None, "rotation": None, "reason": "Product shifted, move into position"}
    source = np.float32([frame_points[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    target = np.float32([reference_points[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, config.ORB_RANSAC_REPROJECTION)
    inliers = int(mask.sum()) if mask is not None else 0
    if homography is None or inliers < config.ALIGN_MIN_INLIERS:
        return {"ok": False, "aligned_image": None, "inliers": inliers, "shift": None, "rotation": None, "reason": "Product shifted, move into position"}
    height, width = reference_gray.shape
    center = np.array([[[frame.shape[1] / 2.0, frame.shape[0] / 2.0]]], dtype=np.float32)
    mapped_center = cv2.perspectiveTransform(center, homography)[0, 0]
    shift = float(np.linalg.norm(mapped_center - np.array([width / 2.0, height / 2.0])))
    rotation = abs(float(np.degrees(np.arctan2(homography[1, 0], homography[0, 0]))))
    if shift > min(width, height) * config.ALIGN_MAX_SHIFT_FRACTION or rotation > config.ALIGN_MAX_ROTATION_DEGREES:
        return {"ok": False, "aligned_image": None, "inliers": inliers, "shift": shift, "rotation": rotation, "reason": "Product shifted, move into position"}
    aligned_bgr = cv2.warpPerspective(frame_bgr, homography, (width, height))
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    return {"ok": True, "aligned_image": aligned_rgb, "inliers": inliers, "shift": shift, "rotation": rotation, "reason": ""}
