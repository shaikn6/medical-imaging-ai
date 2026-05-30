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
import os
import sys
import time

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    os.path.join(_ROOT, "model", "xray_model.pth"),
)

_pipeline: InferencePipeline | None = None


def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(CHECKPOINT):
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found at {CHECKPOINT}. "
                       "Run model/trainer.py first.",
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    t0 = time.perf_counter()

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc

    pipeline = get_pipeline()
    result = pipeline.run(pil_image)

    # Encode Grad-CAM overlay as base64 PNG
    buf = io.BytesIO()
    result.overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    processing_ms = (time.perf_counter() - t0) * 1000.0

    # Persist to DB
    record = PredictionRecord(
        filename=file.filename,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities_json=json.dumps(result.probabilities),
        processing_ms=round(processing_ms, 2),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        filename=file.filename,
        predicted_class=result.predicted_class,
        predicted_idx=result.predicted_idx,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_png_b64=b64,
        processing_ms=round(processing_ms, 2),
    )


@app.get("/predictions", response_model=list[PredictionSummary])
def list_predictions(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
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
