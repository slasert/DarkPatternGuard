"""FastAPI inference service for DarkPatternGuard."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.backend.database import get_db, init_db
from app.backend.models import FeedbackRecord, PredictionRecord
from app.backend.models_registry import Prediction, get_registry
from src.robustness.explain import explain as explain_model
from src.robustness.inference import (
    ArtifactUnavailable,
    from_registry,
    validate_probabilities,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger("darkpatternguard.api")


class Metrics:
    """In-process Prometheus-style request counters."""

    requests = 0
    errors = 0
    total_latency_seconds = 0.0
    latencies_ms: list[float] = []


class TextRequest(BaseModel):
    """Single text prediction request."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    privacy_flag: bool = False

    @field_validator("text")
    @classmethod
    def validate_length(cls, value: str) -> str:
        if len(value.split()) > 500:
            raise ValueError("text must contain at most 500 words")
        return value


class BatchRequest(BaseModel):
    """Batch prediction request capped for interactive serving."""

    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=50)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip() or len(value.split()) > 500:
                raise ValueError("each text must be non-empty and at most 500 words")
        return values


class ExplainRequest(TextRequest):
    """Explain one actual loaded model, optionally compare a controlled edit."""

    model: str | None = None
    perturbed_text: str | None = Field(default=None, max_length=4000)


class FeedbackRequest(BaseModel):
    """User correction attached to a saved prediction."""

    model_config = ConfigDict(extra="forbid")
    prediction_id: int = Field(gt=0)
    user_label: str
    comment: str | None = None

    @field_validator("user_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in ("Urgency", "Scarcity", "Social Proof", "Normal"):
            raise ValueError("user_label must be one of the four project labels")
        return value


def _prediction_payload(
    predictions: dict[str, Prediction], ensemble: Prediction
) -> dict:
    return {
        "label": ensemble.label,
        "confidence_percent": round(ensemble.probabilities[ensemble.label] * 100, 2),
        "per_model": {
            name: {
                "label": prediction.label,
                "probabilities": prediction.probabilities,
                "top_contributing_words": prediction.top_words,
            }
            for name, prediction in predictions.items()
        },
        "calibrated_ensemble": {
            "label": ensemble.label,
            "probabilities": ensemble.probabilities,
        },
        "top_contributing_words": ensemble.top_words,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the serving registry once during application startup."""
    try:
        get_registry().load()
    except (OSError, ValueError, RuntimeError) as error:
        # Keep docs/history and actionable model errors available without
        # substituting heuristic predictions for missing benchmark weights.
        logger.error("trained_model_unavailable: %s", error)
    init_db()
    yield


app = FastAPI(
    title="DarkPatternGuard API",
    version="0.1.0",
    description="CPU inference API for dark-pattern classification.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
results_path = Path("results")
if results_path.is_dir():
    app.mount("/static/results", StaticFiles(directory=results_path), name="results")


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    Metrics.requests += 1
    try:
        response = await call_next(request)
    except Exception:
        Metrics.errors += 1
        logger.exception("request_failed", extra={"request_id": request_id})
        raise
    finally:
        elapsed = time.perf_counter() - started
        Metrics.total_latency_seconds += elapsed
        Metrics.latencies_ms.append(elapsed * 1000)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return response


def _predict_and_save(payload: TextRequest, db: Session) -> dict:
    """Classify and persist one text using an explicit database session."""
    try:
        registry = get_registry()
        predictions = registry.predict(payload.text)
        ensemble = registry.ensemble(predictions)
        record = PredictionRecord(
            text_hash=hashlib.sha256(payload.text.encode("utf-8")).hexdigest(),
            text=None if payload.privacy_flag else payload.text,
            per_model_probs={
                name: prediction.probabilities
                for name, prediction in predictions.items()
            },
            final_label=ensemble.label,
            confidence=ensemble.probabilities[ensemble.label],
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        response = _prediction_payload(predictions, ensemble)
        response["prediction_id"] = record.id
        return response
    except OSError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trained model unavailable: {error}. Restore the original "
                "trained weights and local tokenizer cache."
            ),
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="prediction failed") from error


@app.post("/predict")
def predict(payload: TextRequest, db: Session = Depends(get_db)):  # noqa: B008
    """Classify one e-commerce text."""
    return _predict_and_save(payload, db)


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest, db: Session = Depends(get_db)):  # noqa: B008
    """Classify up to 50 texts."""
    return {
        "predictions": [
            _predict_and_save(TextRequest(text=text), db) for text in payload.texts
        ]
    }


@app.post("/explain")
def explain_prediction(payload: ExplainRequest):
    """Word-level attribution of a real model; no text is persisted here."""
    try:
        model = from_registry(get_registry(), payload.model)
        result = explain_model(model, payload.text)
        if payload.perturbed_text is not None:
            if (
                not payload.perturbed_text.strip()
                or len(payload.perturbed_text.split()) > 80
            ):
                raise ValueError("Perturbed text must contain 1 to 80 words")
            probabilities = validate_probabilities(
                model.probabilities([payload.perturbed_text]), 1
            )[0]
            labels = list(result["probabilities"])
            target = labels.index(result["label"])
            predicted = int(probabilities.argmax())
            result["comparison"] = {
                "text": payload.perturbed_text,
                "label": labels[predicted],
                "confidence": float(probabilities[predicted]),
                "probabilities": dict(zip(labels, probabilities.tolist(), strict=True)),
                "prediction_changed": predicted != target,
                "original_class_probability_delta": float(
                    probabilities[target] - result["confidence"]
                ),
                "note": (
                    "A changed prediction is not necessarily an error "
                    "when meaning changes."
                ),
            }
        return result
    except (ArtifactUnavailable, OSError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/history")
def history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),  # noqa: B008
):
    """Return paginated saved predictions without exposing opted-out text."""
    offset = (page - 1) * page_size
    records = db.scalars(
        select(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
        .offset(offset)
        .limit(page_size)
    ).all()
    total = db.scalar(select(func.count(PredictionRecord.id))) or 0
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [
            {
                "id": record.id,
                "created_at": record.created_at,
                "text_hash": record.text_hash,
                "text": record.text,
                "final_label": record.final_label,
                "confidence": record.confidence,
                "per_model_probs": record.per_model_probs,
            }
            for record in records
        ],
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):  # noqa: B008
    """Save a user correction for an existing prediction."""
    if db.get(PredictionRecord, payload.prediction_id) is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    record = FeedbackRecord(
        prediction_id=payload.prediction_id,
        user_label=payload.user_label,
        comment=payload.comment,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "id": record.id,
        "prediction_id": record.prediction_id,
        "user_label": record.user_label,
        "comment": record.comment,
        "created_at": record.created_at,
    }


@app.get("/stats")
def stats(db: Session = Depends(get_db)):  # noqa: B008
    """Return label distribution, confidence, and feedback agreement."""
    distribution = dict(
        db.execute(
            select(
                PredictionRecord.final_label, func.count(PredictionRecord.id)
            ).group_by(PredictionRecord.final_label)
        ).all()
    )
    average_confidence = db.scalar(select(func.avg(PredictionRecord.confidence)))
    feedback_total = db.scalar(select(func.count(FeedbackRecord.id))) or 0
    feedback_agreement = (
        db.scalar(
            select(func.count(FeedbackRecord.id))
            .join(
                PredictionRecord,
                FeedbackRecord.prediction_id == PredictionRecord.id,
            )
            .where(FeedbackRecord.user_label == PredictionRecord.final_label)
        )
        or 0
    )
    return {
        "label_distribution": distribution,
        "average_confidence": float(average_confidence or 0.0),
        "feedback": {
            "total": feedback_total,
            "agreement": feedback_agreement,
            "agreement_rate": (
                float(feedback_agreement / feedback_total) if feedback_total else 0.0
            ),
        },
    }


@app.get("/models")
def models():
    """Return active model names, versions, and saved Macro-F1 scores."""
    try:
        registry = get_registry()
        return {"models": registry.model_metadata()}
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503, detail=f"Trained model unavailable: {error}"
        ) from error


@app.get("/health")
def health():
    """Return service health and active model count."""
    try:
        registry = get_registry()
        return {"status": "ok", "active_models": len(registry.active_models)}
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=503, detail=f"Trained model unavailable: {error}"
        ) from error


@app.get("/metrics")
def metrics():
    """Return Prometheus-style counters."""
    average = Metrics.total_latency_seconds / max(1, Metrics.requests)
    ordered = sorted(Metrics.latencies_ms)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95)) if ordered else 0
    p95 = ordered[p95_index] if ordered else 0.0
    return (
        "# TYPE darkpatternguard_requests_total counter\n"
        f"darkpatternguard_requests_total {Metrics.requests}\n"
        "# TYPE darkpatternguard_errors_total counter\n"
        f"darkpatternguard_errors_total {Metrics.errors}\n"
        "# TYPE darkpatternguard_request_latency_seconds gauge\n"
        f"darkpatternguard_request_latency_seconds {average:.6f}\n"
        "# TYPE darkpatternguard_request_latency_p95_ms gauge\n"
        f"darkpatternguard_request_latency_p95_ms {p95:.3f}\n"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=False)
