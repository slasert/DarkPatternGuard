"""Strict adapters around the actual Phase 1 serving artifacts; never train."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.eval.metrics import LABELS


class ArtifactUnavailable(RuntimeError):
    """A genuine, identifiable trained model cannot be used."""


class TextModel(Protocol):
    """Shared interface for evaluation and explanations of one frozen model."""

    identity: dict[str, Any]

    def probabilities(self, texts: list[str]) -> np.ndarray: ...


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probabilities(values: Any, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (count, len(LABELS))
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Model probabilities must be finite four-class distributions")
    return values


@dataclass
class ServingModel:
    """Use the same tokenizer, truncation and prediction as the serving model."""

    artifact: Any
    identity: dict[str, Any]

    def probabilities(self, texts: list[str]) -> np.ndarray:
        rows = [self.artifact.predict(text).probabilities for text in texts]
        return validate_probabilities(
            [[row[label] for label in LABELS] for row in rows], len(texts)
        )


def from_registry(registry: Any, name: str | None = None) -> ServingModel:
    """Refuse Phase 1's keyword fallbacks and unidentified test doubles."""
    from app.backend.models_registry import _TransformerArtifact
    from src.robustness.reference import ReferenceModel

    try:
        registry.load()
    except (OSError, ValueError, RuntimeError) as error:
        raise ArtifactUnavailable(
            f"Trained weights unavailable or invalid: {error}. Restore the original "
            "models_out/<model>/seed<seed>.weights.pt and its local Hugging Face "
            "tokenizer/base snapshot. Saved predictions are not trained weights."
        ) from error
    candidates = registry._models
    if name is None:
        if len(candidates) != 1:
            raise ArtifactUnavailable("Select one actual trained model for attribution")
        name = next(iter(candidates))
    artifact = candidates.get(name)
    if isinstance(artifact, ReferenceModel):
        return artifact
    if not isinstance(artifact, _TransformerArtifact) or not artifact.identity:
        raise ArtifactUnavailable(
            f"{name}: no identifiable trained transformer loaded; "
            "heuristic predictions cannot be used for Robustness evaluation"
        )
    return ServingModel(artifact, artifact.identity.copy())


def load_checkpoint(name: str, seed: int, models_dir: Path) -> ServingModel:
    """Explicit model/seed selection, without selecting on challenge labels."""
    from app.backend.models_registry import TRANSFORMER_MODELS, ModelRegistry
    from src.robustness.reference import REFERENCE_MODELS, ReferenceModel

    if name in REFERENCE_MODELS:
        return ReferenceModel(models_dir / "reference" / name / f"seed{seed}.json")

    if name not in TRANSFORMER_MODELS:
        raise ArtifactUnavailable(
            f"Unsupported artifact {name!r}. Phase 1 exposes trained BERT/DeBERTa "
            "loaders only; classical/CNN/BiLSTM loaders are absent in this checkout."
        )
    registry = ModelRegistry(models_dir=models_dir)
    try:
        artifact = registry._load_transformer(name, seed)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        raise ArtifactUnavailable(
            f"Cannot load {name}/seed{seed}: {error}. Restore original trained "
            "weights and the matching locally cached tokenizer/base model. "
            "Evaluation never retrains models or downloads substitute weights."
        ) from error
    registry._models = {name: artifact}
    registry._loaded = True
    return from_registry(registry, name)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint actual token IDs and special-token settings, not just a name."""
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map,
        "init_kwargs": tokenizer.init_kwargs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
