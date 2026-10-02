"""Fit from a good video and replay it through the live inspect API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config

JPEG_QUALITY = 85  # Match the browser canvas.toDataURL('image/jpeg', 0.85) setting.


def encode_frame(frame_bgr: Any) -> bytes:
    """Encode a decoded video frame using the browser JPEG quality setting."""
    encoded, data = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not encoded:
        raise RuntimeError("Could not JPEG-encode video frame")
    return data.tobytes()


def video_metadata(video_path: Path) -> tuple[int, float]:
    """Return readable frame count and frame rate for a video file."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    if count < 2:
        raise SystemExit("Video must contain at least two frames")
    return count, fps


def sample_frames(video_path: Path, sample_count: int) -> list[bytes]:
    """Sample evenly spaced frames, encoded like browser live JPEG frames."""
    total, _ = video_metadata(video_path)
    indices = set(int(value) for value in np.linspace(0, total - 1, min(sample_count, total)))
    capture = cv2.VideoCapture(str(video_path))
    frames: list[bytes] = []
    index = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            frames.append(encode_frame(frame))
        index += 1
    capture.release()
    if len(frames) < 2:
        raise SystemExit("Could not sample at least two frames from the video")
    return frames


def post_multipart(base_url: str, path: str, fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> dict[str, Any]:
    """Send multipart form data and return the decoded JSON response."""
    boundary = f"----VisionQC{uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
            value.encode(),
            b"\r\n",
        ))
    for name, filename, payload in files:
        chunks.extend((
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode(),
            payload,
            b"\r\n",
        ))
    chunks.append(f"--{boundary}--\r\n".encode())
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=1800.0) as response:
            return json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(f"{path} failed ({error.code}): {error.read().decode(errors='replace')}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to VisionQC API: {error.reason}") from error


def inspect_video(base_url: str, video_path: Path, fps: float, replay_fps: float) -> dict[str, Any]:
    """Send video frames to /inspect at the requested rate and count verdicts."""
    frame_stride = max(1, round(fps / replay_fps))
    capture = cv2.VideoCapture(str(video_path))
    counts = {"PASS": 0, "FAIL": 0, "RECAPTURE": 0}
    total = 0
    index = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_stride == 0:
            try:
                response = post_multipart(
                    base_url,
                    "/inspect",
                    {"source": "video"},
                    [("frame", f"frame_{index:06d}.jpg", encode_frame(frame))],
                )
            except RuntimeError:
                capture.release()
                raise
            verdict = response["verdict"]
            counts[verdict] += 1
            total += 1
        index += 1
    capture.release()
    if not total:
        raise RuntimeError("No frames were replayed through /inspect")
    return {"counts": counts, "total": total, "rates_pct": {key.lower(): value * 100.0 / total for key, value in counts.items()}}


def main() -> None:
    """Fit through /fit, replay through /inspect, and report requested metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a good-only video clip")
    parser.add_argument("--frames", type=int, default=config.FIT_MAX_IMAGES, help="Number of evenly spaced frames used for fit")
    parser.add_argument("--replay-fps", type=float, default=3.0, help="Inspection replay rate in frames per second")
    parser.add_argument("--api", default="http://localhost:8000", help="VisionQC API base URL")
    args = parser.parse_args()
    if args.frames < 2 or args.replay_fps <= 0:
        raise SystemExit("--frames must be at least 2 and --replay-fps must be positive")
    video_path = args.video.resolve()
    total_frames, video_fps = video_metadata(video_path)
    fit_frames = sample_frames(video_path, args.frames)
    fitted = post_multipart(
        args.api,
        "/fit",
        {},
        [("frames", f"fit_{index:03d}.jpg", frame) for index, frame in enumerate(fit_frames)],
    )
    replay = inspect_video(args.api, video_path, video_fps, args.replay_fps)
    print(f"Video: {video_path}")
    print(f"Frames in clip: {total_frames}; video fps: {video_fps:.2f}")
    print(f"Fit samples: {fitted['fit']['images']}; memory patches: {fitted['fit']['patches']}")
    print(f"Brightness range: {fitted['capture_calibration']['brightness_min']:.2f} - {fitted['capture_calibration']['brightness_max']:.2f}")
    print(f"Center-crop blur threshold: {fitted['capture_calibration']['sharpness_min']:.4f}")
    print(f"Leave-one-out good-score maximum: {fitted['calibration']['max']:.4f}")
    print(f"Suggested threshold (max x {config.THRESHOLD_MARGIN:.2f}): {fitted['suggested_threshold']:.4f}")
    print(f"Replay via /inspect: {replay['total']} frames")
    print(f"Recapture: {replay['rates_pct']['recapture']:.2f}%")
    print(f"Pass: {replay['rates_pct']['pass']:.2f}%")
    print(f"Fail: {replay['rates_pct']['fail']:.2f}%")
    print(f"Counts: {replay['counts']}")


if __name__ == "__main__":
    main()
