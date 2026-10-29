"""JSON-only frozen TF-IDF models for the newly authorized reference run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.eval.metrics import LABELS
from src.robustness.inference import ArtifactUnavailable, sha256, validate_probabilities

REFERENCE_MODELS = ("reference_logistic_regression", "reference_naive_bayes")


class ReferenceModel:
    """Frozen linear classifier; artifacts contain data, never executable pickle."""

    def __init__(self, path: Path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["schema"] != 1 or payload["labels"] != list(LABELS):
                raise ValueError("Unsupported schema or label mapping")
            if payload["name"] not in REFERENCE_MODELS:
                raise ValueError("Unsupported reference model")
            vocab = payload["vocabulary"]
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"vocabulary": vocab, "idf": payload["idf"]}, sort_keys=True
                ).encode()
            ).hexdigest()
            if fingerprint != payload["vectorizer_sha256"]:
                raise ValueError("Vectorizer fingerprint mismatch")
            if sorted(vocab.values()) != list(range(len(vocab))):
                raise ValueError("Vocabulary indices must be contiguous and unique")
            self.vectorizer = TfidfVectorizer(
                vocabulary=vocab,
                ngram_range=(1, 2),
                sublinear_tf=True,
                lowercase=True,
                strip_accents="unicode",
            )
            self.vectorizer.idf_ = np.asarray(payload["idf"], dtype=float)
            self.weights = np.asarray(payload["weights"], dtype=float)
            self.bias = np.asarray(payload["bias"], dtype=float)
            if (
                self.weights.shape != (4, len(vocab))
                or self.bias.shape != (4,)
                or self.vectorizer.idf_.shape != (len(vocab),)
                or not np.isfinite(self.weights).all()
                or not np.isfinite(self.bias).all()
                or not np.isfinite(self.vectorizer.idf_).all()
            ):
                raise ValueError("Invalid trained coefficient or IDF dimensions/values")
            self.identity = {
                "name": payload["name"],
                "seed": payload["seed"],
                "checkpoint_sha256": sha256(path),
                "tokenizer_sha256": payload["vectorizer_sha256"],
                "labels": list(LABELS),
                "device": "cpu",
                "max_length": None,
                "training_provenance": (
                    "new authorized reference run; not recovered Phase 1 weights"
                ),
                "training": payload["training"],
            }
            self.validation_macro_f1 = float(payload["validation_macro_f1"])
            self.path = path
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactUnavailable(
                f"Invalid reference artifact {path}: {error}"
            ) from error

    def probabilities(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        logits = np.asarray(matrix @ self.weights.T) + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return validate_probabilities(probabilities, len(texts))

    def predict(self, text: str):
        from app.backend.models_registry import Prediction

        values = self.probabilities([text])[0]
        return Prediction(
            LABELS[int(values.argmax())],
            dict(zip(LABELS, values.tolist(), strict=True)),
            [],
        )

    def feature_contributions(self, text: str) -> dict:
        """Exact additive class-logit terms, not additive probability effects."""
        row = self.vectorizer.transform([text]).tocsr()
        probabilities = self.probabilities([text])[0]
        target = int(probabilities.argmax())
        names = self.vectorizer.get_feature_names_out()
        values = row.data * self.weights[target, row.indices]
        features = [
            {"feature": str(names[index]), "score": float(value)}
            for index, value in zip(row.indices, values, strict=True)
        ]
        features.sort(key=lambda item: (-abs(item["score"]), item["feature"]))
        return {
            "target_class": LABELS[target],
            "bias": float(self.bias[target]),
            "logit": float(values.sum() + self.bias[target]),
            "features": features,
            "method": "exact_tfidf_linear_logit_contributions",
            "description": "TF-IDF value × learned class coefficient, plus intercept. "
            "All listed features sum exactly to the target logit. These are not "
            "probability contributions or causal evidence; n-grams can overlap.",
        }
