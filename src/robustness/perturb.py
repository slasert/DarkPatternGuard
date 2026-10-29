"""Metamorphic held-out evaluation; semantic edits remain descriptive only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import LABELS, compute_metrics, save_confusion_matrix
from src.robustness.inference import TextModel, sha256, validate_probabilities


def transpose_one(text: str) -> str:
    """One deterministic internal character swap; meaning preservation unverified."""
    match = re.search(r"\b[a-zA-Z]{5,}\b", text)
    if match is None:
        return text
    word = match.group()
    offset = next((i for i in range(1, len(word) - 2) if word[i] != word[i + 1]), None)
    if offset is None:
        return text
    changed = word[:offset] + word[offset + 1] + word[offset] + word[offset + 2 :]
    return text[: match.start()] + changed + text[match.end() :]


def run_perturbations(
    model: TextModel, test_path: Path, output: Path, seed=1729
) -> dict:
    frame = pd.read_csv(test_path)
    texts = frame["text"].tolist()
    original = validate_probabilities(model.probabilities(texts), len(texts))
    original_labels = np.asarray(LABELS)[original.argmax(axis=1)]
    transformations = {
        "uppercase": ([text.upper() for text in texts], True),
        "expanded_whitespace": (["  \t".join(text.split()) for text in texts], True),
        "single_internal_typo": ([transpose_one(text) for text in texts], False),
    }
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "model": model.identity,
        "test_sha256": sha256(test_path),
        "conditions": {},
        "note": "Case/whitespace scores inherit held-out labels under explicit "
        "surface-invariance assumptions; they are not independent human annotation. "
        "Typo flips are descriptive, not automatically errors. "
        "No challenge labels used.",
    }
    for name, (variants, invariant) in transformations.items():
        probabilities = validate_probabilities(
            model.probabilities(variants), len(texts)
        )
        predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        changed = np.asarray([a != b for a, b in zip(texts, variants, strict=True)])
        flips = predictions != original_labels
        details = []
        for index, text in enumerate(texts):
            details.append(
                {
                    "id": index,
                    "original": text,
                    "perturbed": variants[index],
                    "text_changed": bool(changed[index]),
                    "original_prediction": original_labels[index],
                    "perturbed_prediction": predictions[index],
                    "prediction_changed": bool(flips[index]),
                    "total_variation": float(
                        abs(probabilities[index] - original[index]).sum() / 2
                    ),
                }
            )
        condition = {
            "count": len(texts),
            "changed_texts": int(changed.sum()),
            "prediction_flips": int(flips.sum()),
            "flip_rate": float(flips.mean()),
            "flip_rate_changed_only": (
                float(flips[changed].mean()) if changed.any() else None
            ),
            "inherited_label_assumption": invariant,
        }
        if invariant:
            metrics = compute_metrics(
                frame["label"].tolist(), predictions, probabilities, seed=seed
            )
            condition["macro_f1_inherited_labels"] = metrics["macro_f1"]
            (output / f"{name}-metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            save_confusion_matrix(
                frame["label"].tolist(),
                predictions,
                output / f"{name}-confusion.png",
                name,
            )
        summary["conditions"][name] = condition
        (output / f"{name}-pairs.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
