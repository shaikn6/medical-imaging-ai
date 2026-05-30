"""
FastAPI application — chest X-ray pathology detection.

Endpoints:
    POST /predict          Upload an image → JSON prediction + base64 Grad-CAM
    GET  /predictions      List recent prediction records
    GET  /health           Health check
    GET  /classes          Return supported class names

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.database import get_db, init_db      # noqa: E402
from api.models import PredictionRecord       # noqa: E402
from model.inference import InferencePipeline # noqa: E402
from model.cnn_classifier import CLASSES      # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Imaging AI",
    description="Chest X-ray pathology detection with Grad-CAM explainability",
    version="1.1.0",
)

# ---------------------------------------------------------------------------
# Security: restrict CORS to specific origins rather than wildcard "*".
# Override via ALLOWED_ORIGINS env var (comma-separated list).
# ---------------------------------------------------------------------------
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# ---------------------------------------------------------------------------
# Security: upload limits
# ---------------------------------------------------------------------------
# Maximum upload size: 10 MB for standard images; real DICOM can be larger
# but 10 MB is a safe ceiling for single-slice X-ray images.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

# Allowed MIME types (content_type check only — PIL validates actual bytes)
_ALLOWED_CONTENT_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",   # Pillow can decode; GIF falls back gracefully
    "image/webp",
    "image/bmp",
    "image/tiff",
})

CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    os.path.join(_ROOT, "model", "xray_model.pth"),
)

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(CHECKPOINT):
            # Do NOT expose the filesystem path in the response body.
            logger.error("Model checkpoint not found: %s", CHECKPOINT)
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint unavailable. "
                       "Contact the administrator or run the training script.",
            )
        _pipeline = InferencePipeline(CHECKPOINT)
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    filename: str | None
    predicted_class: str
    predicted_idx: int
    confidence: float
    probabilities: dict[str, float]
    gradcam_png_b64: str        # base64-encoded PNG of the Grad-CAM overlay
    processing_ms: float


class PredictionSummary(BaseModel):
    id: int
    filename: str | None
    predicted_class: str
    confidence: float
    created_at: str


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": os.path.exists(CHECKPOINT)}


@app.get("/classes")
def list_classes():
    return {"classes": CLASSES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a grayscale chest X-ray (PNG/JPEG) and receive:
    - Predicted pathology class and confidence
    - Per-class probability breakdown
    - Grad-CAM heatmap overlay (base64 PNG)
    """
    # --- Content-type validation (defense-in-depth; PIL validates actual bytes) ---
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PNG or JPEG image.",
        )

    t0 = time.perf_counter()

    # --- File-size limit: read in chunks to avoid buffering a huge malicious upload ---
    raw_chunks: list[bytes] = []
    bytes_read = 0
    chunk_size = 65536  # 64 KiB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024*1024)} MB).",
            )
        raw_chunks.append(chunk)
    raw = b"".join(raw_chunks)

    # --- Decode image bytes (PIL validates magic bytes, not just MIME) ---
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        # Do NOT echo exception details back — they can expose PIL internals / paths
        raise HTTPException(status_code=422, detail="Cannot decode image file.") from None

    pipeline = get_pipeline()
    result = pipeline.run(pil_image)

    # Encode Grad-CAM overlay as base64 PNG
    buf = io.BytesIO()
    result.overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    processing_ms = (time.perf_counter() - t0) * 1000.0

    # --- Sanitise the filename: keep only the basename to prevent path traversal ---
    raw_filename = file.filename or ""
    safe_filename = os.path.basename(raw_filename)[:256] or None

    # Persist to DB (safe_filename only — never the raw user-supplied path)
    record = PredictionRecord(
        filename=safe_filename,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities_json=json.dumps(result.probabilities),
        processing_ms=round(processing_ms, 2),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        filename=safe_filename,
        predicted_class=result.predicted_class,
        predicted_idx=result.predicted_idx,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_png_b64=b64,
        processing_ms=round(processing_ms, 2),
    )


@app.get("/predictions", response_model=list[PredictionSummary])
def list_predictions(limit: int = 20, db: Session = Depends(get_db)):
    # Cap the limit to prevent excessively large result sets
    capped_limit = max(1, min(limit, 200))
    rows = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(capped_limit)
        .all()
    )
    return [
        PredictionSummary(
            id=r.id,
            filename=r.filename,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            created_at=str(r.created_at),
        )
        for r in rows
    ]
