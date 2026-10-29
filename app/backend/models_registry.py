"""Lazy CPU model registry for the inference API."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ("Urgency", "Scarcity", "Social Proof", "Normal")
PROBABILITY_COLUMNS = (
    "prob_Urgency",
    "prob_Scarcity",
    "prob_Social_Proof",
    "prob_Normal",
)
TRANSFORMER_MODELS = {
    "finetune_bert": "bert-base-uncased",
    "finetune_deberta": "microsoft/deberta-v3-base",
}


@dataclass
class Prediction:
    """One model prediction and lightweight explanation."""

    label: str
    probabilities: dict[str, float]
    top_words: list[str]


@dataclass
class _TransformerArtifact:
    """A trained CPU transformer and its tokenizer."""

    model: Any
    tokenizer: Any
    identity: dict[str, Any] | None = None

    def predict(self, text: str) -> Prediction:
        """Run inference using the trained model's four-class probabilities."""
        import torch

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return Prediction(
            label=LABELS[int(probabilities.argmax())],
            probabilities=dict(zip(LABELS, probabilities.tolist(), strict=True)),
            top_words=[],
        )


class ModelRegistry:
    """Load serving models once and expose a uniform prediction interface."""

    def __init__(
        self,
        results_dir: Path = Path("results"),
        models_dir: Path = Path("models_out"),
    ) -> None:
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.use_llm = os.getenv("USE_LLM_IN_API", "false").lower() == "true"
        self._loaded = False
        self._models: dict[str, Any] = {}

    def load(self) -> None:
        """Load CPU-only serving components once at application startup."""
        if self._loaded:
            return
        from src.robustness.reference import ReferenceModel

        explicit_reference = os.getenv("DPG_REFERENCE_ARTIFACT")
        if explicit_reference:
            reference = ReferenceModel(Path(explicit_reference))
            self._models = {reference.identity["name"]: reference}
            self._loaded = True
            return
        candidate = self._best_transformer_candidate()
        reference_paths = sorted((self.models_dir / "reference").glob("*/seed*.json"))
        if candidate is not None:
            model_name, seed = candidate
            self._models[model_name] = self._load_transformer(model_name, seed)
        elif reference_paths:
            references = [ReferenceModel(path) for path in reference_paths]
            reference = max(
                references,
                key=lambda item: (
                    item.validation_macro_f1,
                    item.identity["name"],
                    -item.identity["seed"],
                ),
            )
            self._models = {reference.identity["name"]: reference}
        elif (self.results_dir / "leaderboard.csv").is_file():
            raise FileNotFoundError(
                "A benchmark leaderboard exists, but no trained transformer "
                "checkpoint with validation predictions is available to serve."
            )
        else:
            self._models = {
                "logistic_regression": self._load_optional("logistic_regression"),
                "lightgbm": self._load_optional("lightgbm"),
                "distilbert": self._load_optional("distilbert"),
            }
            if self.use_llm:
                self._models["gpt-4o-mini"] = self._load_optional("llm_gpt4o_mini")
        self._loaded = True

    def _best_transformer_candidate(self) -> tuple[str, int] | None:
        """Select the seed/model with the highest validation Macro-F1."""
        candidates: list[tuple[float, str, int]] = []
        for model_name in TRANSFORMER_MODELS:
            model_results = self.results_dir / model_name
            for seed_dir in sorted(model_results.glob("seed*")):
                if not seed_dir.name.startswith("seed"):
                    continue
                try:
                    seed = int(seed_dir.name.removeprefix("seed"))
                except ValueError:
                    continue
                validation_path = seed_dir / "val_preds.csv"
                weights_path = self.models_dir / model_name / f"seed{seed}.weights.pt"
                if not validation_path.is_file():
                    continue
                if not weights_path.is_file():
                    # Select among artifacts actually restored on this host.
                    # Saved predictions alone are never serving candidates.
                    continue
                frame = pd.read_csv(validation_path)
                required_columns = {"true_label", *PROBABILITY_COLUMNS}
                if frame.empty or not required_columns.issubset(frame.columns):
                    raise ValueError(
                        f"Invalid validation prediction file: {validation_path}"
                    )
                probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
                if not np.isfinite(probabilities).all():
                    raise ValueError(
                        f"Non-finite validation probabilities: {validation_path}"
                    )
                predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
                score = float(
                    f1_score(
                        frame["true_label"],
                        predicted,
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    )
                )
                candidates.append((score, model_name, seed))
        if not candidates:
            return None
        _, model_name, seed = max(
            candidates, key=lambda item: (item[0], item[1], -item[2])
        )
        return model_name, seed

    def _load_transformer(self, model_name: str, seed: int) -> _TransformerArtifact:
        """Load a locally cached base model and its D:-backed trained weights."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.robustness.inference import sha256, tokenizer_fingerprint

        weights_path = self.models_dir / model_name / f"seed{seed}.weights.pt"
        if not weights_path.is_file() or weights_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty trained checkpoint: {weights_path}"
            )
        checkpoint_hash = sha256(weights_path)

        model_id = TRANSFORMER_MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={label: index for index, label in enumerate(LABELS)},
            local_files_only=True,
        )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except (EOFError, pickle.UnpicklingError, TypeError) as error:
            raise ValueError(
                f"Invalid trained checkpoint {weights_path}: {error}"
            ) from error
        if sha256(weights_path) != checkpoint_hash:
            raise ValueError(
                "Checkpoint changed while loading; retry with frozen files"
            )
        model.eval()
        return _TransformerArtifact(
            model=model,
            tokenizer=tokenizer,
            identity={
                "name": model_name,
                "seed": seed,
                "base_model": model_id,
                "checkpoint_sha256": checkpoint_hash,
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                "labels": list(LABELS),
                "max_length": 256,
                "device": "cpu",
                "label_mapping_source": "Phase 1 registry LABELS convention",
                "training_provenance": "operator-supplied; hash identifies bytes only",
            },
        )

    def _load_optional(self, name: str) -> Any:
        """Return a persisted model when available, otherwise a safe fallback."""
        artifact = self.results_dir / name / "serving_model.json"
        if artifact.is_file():
            return json.loads(artifact.read_text(encoding="utf-8"))
        return None

    @property
    def active_models(self) -> list[str]:
        """Return active registry model names."""
        self.load()
        return list(self._models)

    def predict(self, text: str) -> dict[str, Prediction]:
        """Predict with each active model without reloading state."""
        self.load()
        from src.robustness.reference import ReferenceModel

        predictions = {}
        for name in self._models:
            model = self._models[name]
            predictions[name] = (
                model.predict(text)
                if isinstance(model, _TransformerArtifact | ReferenceModel)
                else self._fallback_prediction(text, name)
            )
        return predictions

    def _fallback_prediction(self, text: str, model_name: str) -> Prediction:
        """Provide deterministic serving behavior before trained artifacts exist."""
        lowered = text.lower()
        scores = np.ones(len(LABELS), dtype=float)
        if any(word in lowered for word in ("hurry", "expires", "countdown", "now")):
            scores[0] += 3
        if any(word in lowered for word in ("only", "left", "stock", "limited")):
            scores[1] += 3
        if any(
            word in lowered for word in ("people", "customers", "bestseller", "reviews")
        ):
            scores[2] += 3
        probabilities = scores / scores.sum()
        top_words = [
            word
            for word in ("hurry", "expires", "only", "left", "people", "reviews")
            if word in lowered
        ]
        return Prediction(
            label=LABELS[int(probabilities.argmax())],
            probabilities=dict(zip(LABELS, probabilities.tolist(), strict=True)),
            top_words=top_words[:5],
        )

    def ensemble(self, predictions: dict[str, Prediction]) -> Prediction:
        """Average active model probabilities into the serving ensemble."""
        if not predictions:
            raise RuntimeError("No active models are loaded")
        probabilities = np.mean(
            [
                [prediction.probabilities[label] for label in LABELS]
                for prediction in predictions.values()
            ],
            axis=0,
        )
        words = sorted(
            {
                word
                for prediction in predictions.values()
                for word in prediction.top_words
            }
        )
        return Prediction(
            label=LABELS[int(probabilities.argmax())],
            probabilities=dict(zip(LABELS, probabilities.tolist(), strict=True)),
            top_words=words[:5],
        )

    def model_metadata(self) -> list[dict[str, Any]]:
        """Read model Macro-F1 metadata from leaderboard files when present."""
        self.load()
        reference_metadata = [
            {
                "model": name,
                "version": "reference-retrained",
                "macro_f1": None,
                "validation_macro_f1": model.validation_macro_f1,
                "active": True,
                "provenance": model.identity["training_provenance"],
            }
            for name, model in self._models.items()
            if hasattr(model, "validation_macro_f1")
        ]
        leaderboard_json = self.results_dir / "leaderboard.json"
        if leaderboard_json.is_file():
            payload = json.loads(leaderboard_json.read_text(encoding="utf-8"))
            return (
                payload if isinstance(payload, list) else payload.get("models", [])
            ) + reference_metadata
        leaderboard_csv = self.results_dir / "leaderboard.csv"
        if leaderboard_csv.is_file():
            import pandas as pd

            frame = pd.read_csv(leaderboard_csv)
            return [
                {
                    "model": row.get("model", ""),
                    "version": "trained",
                    "macro_f1": row.get("mean_macro_f1"),
                    "active": row.get("model", "") in self.active_models,
                }
                for row in frame.to_dict(orient="records")
            ] + reference_metadata
        return [
            {"model": name, "version": "fallback", "macro_f1": None}
            for name in self.active_models
        ]


_registry: ModelRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ModelRegistry:
    """Return the process-wide lazily initialized registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
                _registry.load()
    return _registry


def set_registry(registry: ModelRegistry | None) -> None:
    """Replace the singleton for tests and controlled application embedding."""
    global _registry
    _registry = registry
