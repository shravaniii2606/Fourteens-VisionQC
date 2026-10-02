"""FastAPI application for local VisionQC setup and inspections."""

from __future__ import annotations

import base64
import io
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile
from dotenv import load_dotenv

from . import analytics, capture_check, chat, config, db, decision
from .patchcore import PatchCore

model: PatchCore | None = None
model_error: str | None = None


def _false_rejection_pct(calibration: dict[str, Any], threshold: float) -> float | None:
    """Return the share of leave-one-out good scores above the active cutoff."""
    scores = calibration.get("good_scores", [])
    if not scores:
        return None
    return round(sum(float(score) > threshold for score in scores) / len(scores) * 100.0, 2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize SQLite and restore an existing bank without fitting."""
    global model, model_error
    db.connect().close()
    bank_path = config.DATA_DIR / "memory_bank.pt"
    if bank_path.exists():
        try:
            model = PatchCore()
            model_error = None
        except Exception as error:
            model_error = str(error)
    yield


app = FastAPI(title="VisionQC", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ThresholdInput(BaseModel):
    """Supervisor's anomaly decision cutoff."""

    threshold: float = Field(gt=0)


class FeedbackInput(BaseModel):
    """False-alarm feedback confirmation."""

    confirm: bool


class ChatInput(BaseModel):
    """Inspection-data question."""

    question: str = Field(min_length=1, max_length=1000)


class FitInput(BaseModel):
    """JSON fit request for the built-in MVTec category."""

    use_dataset: bool = False


def get_model() -> PatchCore:
    """Return the process-wide model or report its initialization error."""
    global model, model_error
    if model is None:
        try:
            model = PatchCore()
            model_error = None
        except Exception as error:
            model_error = str(error)
            raise HTTPException(status_code=503, detail=f"Could not load pretrained model: {error}") from error
    return model


def _decode_image(raw: bytes) -> np.ndarray:
    """Decode image bytes into a validated RGB uint8 array."""
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_pil(raw: bytes) -> Image.Image:
    """Decode uploaded bytes as RGB with a consistent client error."""
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Invalid image file") from error


@app.post("/fit")
async def fit(request: Request) -> dict[str, Any]:
    """Fit and calibrate from uploads, live JPEG frames, or category/train/good."""
    uploads: list[bytes] = []
    live_frames: list[np.ndarray] = []
    use_dataset = False
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            use_dataset = FitInput(**(await request.json())).use_dataset
        except Exception as error:
            raise HTTPException(status_code=400, detail="Expected {use_dataset: true}") from error
    elif "multipart/form-data" in content_type:
        form = await request.form()
        use_dataset = str(form.get("use_dataset", "false")).lower() in ("true", "1", "yes")
        for entry in form.getlist("files"):
            if isinstance(entry, StarletteUploadFile):
                uploads.append(await entry.read())
        for entry in form.getlist("frames") + form.getlist("frame"):
            if isinstance(entry, StarletteUploadFile):
                live_frames.append(_decode_image(await entry.read()))
    if use_dataset:
        good_dir = config.DATA_DIR / "mvtec" / config.CATEGORY / "train" / "good"
        paths = sorted(path for path in good_dir.glob("*") if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))[:config.FIT_MAX_IMAGES]
        if not paths:
            raise HTTPException(status_code=400, detail=f"No good images found in {good_dir}")
        images = []
        for path in paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(status_code=400, detail=f"Could not read {path.name}") from error
    elif live_frames:
        images = live_frames[:config.FIT_MAX_IMAGES]
    else:
        if not uploads:
            raise HTTPException(status_code=400, detail="Upload good photos or choose Fit from dataset")
        images = [_read_pil(raw) for raw in uploads[:config.FIT_MAX_IMAGES]]
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two good photos")

    active_model = get_model()
    try:
        fit_result = active_model.fit(images)
        calibration = active_model.calibrate(images)
        raw_images = [np.asarray(image, dtype=np.uint8) for image in images]
        capture_stats = capture_check.calibrate_capture(raw_images)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    reference = raw_images[0]
    cv2.imwrite(str(config.DATA_DIR / "reference.jpg"), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
    suggested = max(calibration["max"] * config.THRESHOLD_MARGIN, 1e-6)
    db.set_threshold(suggested)
    db.set_setting("calibration", calibration)
    db.set_setting("fitted_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return {
        "fit": fit_result,
        "calibration": calibration,
        "capture_calibration": capture_stats,
        "suggested_threshold": suggested,
        "threshold": suggested,
        "false_rejection_pct": _false_rejection_pct(calibration, suggested),
    }


@app.get("/settings")
def read_settings() -> dict[str, int | bool]:
    """Return the persisted Settings page controls."""
    return db.get_app_settings()


@app.post("/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, int | bool]:
    """Validate and persist partial Settings page updates."""
    allowed = {"window_n", "cluster_k", "show_seed"}
    unknown = set(payload) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported setting: {sorted(unknown)[0]}")

    current = db.get_app_settings()
    window_n = payload.get("window_n", current["window_n"])
    cluster_k = payload.get("cluster_k", current["cluster_k"])
    if "window_n" in payload and (
        isinstance(window_n, bool) or not isinstance(window_n, int) or not 5 <= window_n <= 200
    ):
        raise HTTPException(status_code=400, detail="window_n must be an integer between 5 and 200.")
    if "cluster_k" in payload and (
        isinstance(cluster_k, bool) or not isinstance(cluster_k, int) or cluster_k < 2
    ):
        raise HTTPException(status_code=400, detail="cluster_k must be an integer of at least 2 and no greater than window_n.")
    if cluster_k > window_n:
        raise HTTPException(status_code=400, detail="cluster_k must be between 2 and window_n.")
    if "show_seed" in payload and not isinstance(payload["show_seed"], bool):
        raise HTTPException(status_code=400, detail="show_seed must be true or false.")

    for key, value in payload.items():
        db.set_setting(key, value)
    return db.get_app_settings()


@app.get("/model_info")
def read_model_info() -> dict[str, Any]:
    """Return model configuration and non-secret chat-key status."""
    load_dotenv(config.ROOT_DIR / ".env")
    fitted = model is not None and model.memory_bank is not None
    return {
        "backbone": config.BACKBONE,
        "category": config.CATEGORY,
        "bank_size": int(model.memory_bank.shape[0]) if fitted and model is not None else 0,
        "fitted_at": db.get_setting("fitted_at"),
        "align_enabled": config.ALIGN_ENABLED,
        "chat_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/reset_log")
def reset_log() -> dict[str, int]:
    """Delete inspection logs, their saved frames, and alerts only."""
    with db.session() as connection:
        saved_files = connection.execute(
            "SELECT frame_path, heatmap_path FROM inspections WHERE frame_path IS NOT NULL OR heatmap_path IS NOT NULL"
        ).fetchall()
        inspections_deleted = connection.execute("DELETE FROM inspections").rowcount
        alerts_deleted = connection.execute("DELETE FROM alerts").rowcount

    frame_root = config.FRAMES_DIR.resolve()
    for row in saved_files:
        for field in ("frame_path", "heatmap_path"):
            if not row[field]:
                continue
            saved_path = Path(row[field]).resolve()
            try:
                saved_path.relative_to(frame_root)
            except ValueError:
                continue
            if saved_path.is_file():
                saved_path.unlink()

    return {
        "inspections_deleted": max(inspections_deleted, 0),
        "alerts_deleted": max(alerts_deleted, 0),
        "rows_deleted": max(inspections_deleted, 0) + max(alerts_deleted, 0),
    }


@app.post("/inspect")
async def inspect(frame: UploadFile = File(...), source: str = Form("webcam")) -> dict[str, Any]:
    """Inspect and log one JPEG frame from a webcam or video file."""
    if source not in ("webcam", "video"):
        raise HTTPException(status_code=400, detail="Source must be webcam or video")
    if model is None and not (config.DATA_DIR / "memory_bank.pt").exists():
        raise HTTPException(status_code=400, detail="Fit the model first")
    active_model = get_model()
    if active_model.memory_bank is None:
        raise HTTPException(status_code=400, detail="Fit the model first")
    rgb = _decode_image(await frame.read())
    try:
        result = decision.decide(rgb, active_model)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail="Fit the model first") from error
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    frame_path = config.FRAMES_DIR / f"{stem}.jpg"
    cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    heatmap_path: Path | None = None
    if result["heatmap_png_base64"]:
        heatmap_path = config.FRAMES_DIR / f"{stem}_heatmap.png"
        heatmap_path.write_bytes(base64.b64decode(result["heatmap_png_base64"]))
    hotspot = result["hotspot"]
    debug = result.get("debug") or {}
    inspection_id = db.add_inspection(
        source=source,
        verdict=result["verdict"],
        reason=result["reason"],
        score=float(result["score"]),
        confidence=float(result["confidence_pct"]),
        hotspot=(hotspot["x"], hotspot["y"]) if hotspot else None,
        frame_path=str(frame_path),
        heatmap_path=str(heatmap_path) if heatmap_path else None,
        brightness=debug.get("brightness"),
        sharpness=debug.get("sharpness"),
        alignment_shift=debug.get("alignment_shift"),
    )
    new_alerts = analytics.repeat_defect_check()
    response = {key: value for key, value in result.items() if key != "aligned_frame"}
    response.update({"id": inspection_id, "source": source, "new_alerts": new_alerts})
    return response


@app.get("/threshold")
def read_threshold() -> dict[str, Any]:
    """Return the active supervisor threshold and calibration interval."""
    calibration = db.get_setting("calibration", {})
    threshold = db.get_threshold()
    return {
        "threshold": threshold,
        "min": float(calibration.get("min", 0.0)),
        "max": float(calibration.get("max", threshold * 2.0)),
        "calibrated": bool(calibration.get("good_scores")),
        "false_rejection_pct": _false_rejection_pct(calibration, threshold),
    }


@app.get("/threshold/suggestion")
def suggest_threshold() -> dict[str, float]:
    """Return the maximum leave-one-out good score with the configured margin."""
    calibration = db.get_setting("calibration", {})
    if not calibration.get("good_scores"):
        raise HTTPException(status_code=400, detail="Fit the model first")
    suggested = max(float(calibration["max"]) * config.THRESHOLD_MARGIN, 1e-6)
    return {"suggested_threshold": suggested}


@app.post("/threshold")
def update_threshold(payload: ThresholdInput) -> dict[str, Any]:
    """Update the supervisor's positive score threshold."""
    db.set_threshold(payload.threshold)
    calibration = db.get_setting("calibration", {})
    return {
        "threshold": payload.threshold,
        "false_rejection_pct": _false_rejection_pct(calibration, payload.threshold),
    }


@app.post("/feedback/{inspection_id}")
def feedback(inspection_id: int, payload: FeedbackInput) -> dict[str, Any]:
    """Add a confirmed false alarm image to the normal bank."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="False-alarm feedback was not confirmed")
    row = db.get_inspection(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    if row["verdict"] != "FAIL":
        raise HTTPException(status_code=400, detail="Only failed inspections can be confirmed as false alarms")
    frame_path = Path(row["frame_path"] or "")
    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise HTTPException(status_code=404, detail="Saved inspection frame is missing")
    active_model = get_model()
    try:
        bank_size = active_model.add_to_bank(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        db.mark_false_alarm(inspection_id)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"id": inspection_id, "bank_size": bank_size}


@app.get("/stats")
def read_stats() -> dict[str, Any]:
    """Return dashboard counts, rates, trends, and drift state."""
    result = analytics.stats()
    calibration = db.get_setting("calibration", {})
    result["drift"] = analytics.drift(calibration.get("mean"))
    result["capture_health"] = result["drift"]["capture_health"]
    result["capture_baseline"] = result["drift"]["capture_baseline"]
    return result


@app.get("/driftguard/status")
def driftguard_status() -> dict[str, Any]:
    """Return the complete DriftGuard rolling-window status."""
    calibration = db.get_setting("calibration", {})
    return analytics.drift(calibration.get("mean"))


@app.post("/driftguard/reset")
def reset_driftguard() -> dict[str, Any]:
    """Start a fresh DriftGuard window without deleting inspection history."""
    reset_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.set_setting("drift_reset_at", reset_at)
    return {"reset_at": reset_at, "minimum_needed": config.DRIFT_MIN_PASSED, "units_used": 0}


@app.get("/history")
def history(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent inspection rows."""
    if limit < 1 or limit > config.HISTORY_MAX_LIMIT:
        raise HTTPException(status_code=400, detail=f"limit must be between 1 and {config.HISTORY_MAX_LIMIT}")
    return db.get_history(limit)


@app.get("/cumulative_heatmap")
def read_cumulative_heatmap() -> dict[str, Any]:
    """Return the latest repeat-defect grid with cell and alert metadata."""
    return analytics.cumulative_heatmap()


@app.post("/repeat-defect/demo")
def seed_repeat_defect_demo() -> dict[str, Any]:
    """Create five same-location failures so the repeat-defect alert is demoable."""
    settings = db.get_app_settings()
    db.set_setting("show_seed", True)
    for index in range(5):
        db.add_inspection(
            source="seed",
            verdict="FAIL",
            reason="Anomaly score exceeds threshold",
            score=2.0 + index * 0.1,
            confidence=0.0,
            hotspot=(0.45, 0.45),
            frame_path=None,
            heatmap_path=None,
        )
    analytics.repeat_defect_check(window=int(settings["window_n"]), required=int(settings["cluster_k"]))
    return analytics.cumulative_heatmap()


@app.get("/alerts")
def read_alerts() -> list[dict[str, Any]]:
    """Return active/recent repeated-defect alerts."""
    return db.get_alerts(config.ALERTS_LIMIT)


@app.post("/chat")
def ask(payload: ChatInput) -> dict[str, str]:
    """Answer a question from SQL inspection records only."""
    return chat.answer(payload.question)


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service/model setup status."""
    return {
        "ok": True,
        "model_loaded": model is not None,
        "fitted": model is not None and model.memory_bank is not None,
        "model_error": model_error,
    }
