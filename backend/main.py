"""FastAPI application for local VisionQC setup and inspections."""

from __future__ import annotations

import base64
import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile

from . import analytics, capture_check, chat, config, db, decision
from .patchcore import PatchCore

model: PatchCore | None = None
model_error: str | None = None


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
    """Fit and calibrate from uploads or category/train/good."""
    uploads: list[bytes] = []
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
    margin = max(calibration["std"] * 0.05, 1e-6)
    suggested = calibration["max"] + margin
    db.set_threshold(suggested)
    db.set_setting("calibration", calibration)
    return {
        "fit": fit_result,
        "calibration": calibration,
        "capture_calibration": capture_stats,
        "suggested_threshold": suggested,
        "threshold": suggested,
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
    inspection_id = db.add_inspection(
        source=source,
        verdict=result["verdict"],
        reason=result["reason"],
        score=float(result["score"]),
        confidence=float(result["confidence_pct"]),
        hotspot=(hotspot["x"], hotspot["y"]) if hotspot else None,
        frame_path=str(frame_path),
        heatmap_path=str(heatmap_path) if heatmap_path else None,
    )
    new_alerts = analytics.repeat_defect_check()
    response = {key: value for key, value in result.items() if key != "aligned_frame"}
    response.update({"id": inspection_id, "source": source, "new_alerts": new_alerts})
    return response


@app.get("/threshold")
def read_threshold() -> dict[str, float]:
    """Return the active supervisor threshold and calibration interval."""
    calibration = db.get_setting("calibration", {})
    return {
        "threshold": db.get_threshold(),
        "min": float(calibration.get("min", 0.0)),
        "max": float(calibration.get("max", db.get_threshold() * 2.0)),
    }


@app.post("/threshold")
def update_threshold(payload: ThresholdInput) -> dict[str, float]:
    """Update the supervisor's positive score threshold."""
    db.set_threshold(payload.threshold)
    return {"threshold": payload.threshold}


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
    return result


@app.get("/history")
def history(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent inspection rows."""
    if limit < 1 or limit > config.HISTORY_MAX_LIMIT:
        raise HTTPException(status_code=400, detail=f"limit must be between 1 and {config.HISTORY_MAX_LIMIT}")
    return db.get_history(limit)


@app.get("/cumulative_heatmap")
def read_cumulative_heatmap() -> dict[str, Any]:
    """Return the latest repeat-defect count grid."""
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
