"""Read-only frozen-model evaluation under original and challenge conditions."""

from __future__ import annotations

import json
import platform
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.benchmark import error_rows
from src.eval.metrics import (
    LABELS,
    compute_metrics,
    save_confusion_matrix,
    save_prediction_csv,
)
from src.robustness.challenge import audit_contamination, eligible, load_challenge
from src.robustness.inference import TextModel, sha256, validate_probabilities

COMPARISON_NOTE = (
    "Phase 1 test texts are naturally sampled, source-group-held-out examples; "
    "The authored challenge is a small, adversarial, manually reviewed set. "
    "Class frequencies, selection and difficulty differ. Scores describe "
    "conditions, not a directly comparable improvement or degradation estimate. "
    "Weights are frozen during evaluation; new reference training is recorded "
    "separately and never uses challenge labels. Paired variants "
    "are dependent; row-bootstrap CIs are exploratory, not significance tests."
)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def configure(seed: int) -> None:
    import torch

    plt.switch_backend("Agg")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def confidence_summary(probabilities: np.ndarray, bins: int) -> dict:
    confidence = probabilities.max(axis=1)
    counts, edges = np.histogram(confidence, bins=np.linspace(0, 1, bins + 1))
    return {
        "count": len(confidence),
        "mean": float(confidence.mean()) if len(confidence) else None,
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }


def pair_changes(
    rows: list[dict], probabilities: np.ndarray, audit: dict
) -> list[dict]:
    by_id = {row["id"]: (index, row) for index, row in enumerate(rows)}
    changes = []
    for index, row in enumerate(rows):
        if not row["parent_id"]:
            continue
        parent_index, parent = by_id[row["parent_id"]]
        usable = not (
            audit["rows"][row["id"]]["excluded"]
            or audit["rows"][parent["id"]]["excluded"]
        )
        target = int(probabilities[parent_index].argmax())
        prediction = int(probabilities[index].argmax())
        reviewed = eligible(row, audit) and eligible(parent, audit)
        expected = row["relation"]
        changes.append(
            {
                "id": row["id"],
                "parent_id": parent["id"],
                "relation": expected,
                "original_label": LABELS[target],
                "perturbed_label": LABELS[prediction],
                "prediction_changed": prediction != target,
                "original_class_probability_delta": float(
                    probabilities[index, target] - probabilities[parent_index, target]
                ),
                "total_variation": float(
                    abs(probabilities[index] - probabilities[parent_index]).sum() / 2
                ),
                "contamination_clean": usable,
                "reviewed_pair": reviewed,
                "invariance_violation": (
                    (prediction != target)
                    if reviewed and expected == "invariant"
                    else None
                ),
                "both_predictions_correct": (
                    (
                        LABELS[target] == parent["label"]
                        and LABELS[prediction] == row["label"]
                    )
                    if reviewed
                    else None
                ),
            }
        )
    return changes


def _condition(
    frame: pd.DataFrame, probabilities: np.ndarray, output: Path, config: dict
) -> dict | None:
    if frame.empty:
        return None
    output.mkdir(parents=True)
    labels = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    # Reuse all existing classification/calibration/bootstrap and matrix code.
    metrics = compute_metrics(
        frame["true_label"].tolist(),
        labels,
        probabilities,
        seed=config["seed"],
        bootstrap_resamples=config["bootstrap_resamples"],
    )
    save_prediction_csv(
        output / "preds.csv",
        frame["id"].tolist(),
        frame["true_label"].tolist(),
        probabilities,
    )
    save_confusion_matrix(
        frame["true_label"].tolist(),
        labels,
        output / "confusion_matrix.png",
        output.name,
    )
    errors = error_rows(frame, labels)
    errors.to_csv(output / "failure_cases.csv", index=False)
    if "category" in frame:
        metrics["categorized_failures"] = {
            category: {
                "evaluated": int((frame["category"] == category).sum()),
                "errors": int((errors["category"] == category).sum()),
            }
            for category in sorted(frame["category"].unique())
        }
    metrics["confidence_distribution"] = confidence_summary(
        probabilities, config["confidence_bins"]
    )
    write_json(output / "metrics.json", metrics)
    return metrics


def evaluate(
    model: TextModel,
    rows: list[dict],
    audit: dict,
    splits: dict[str, Path],
    output: Path,
    config: dict,
) -> dict:
    """Score reviewed clean labels only; retain unreviewed predictions separately."""
    probabilities = validate_probabilities(
        model.probabilities([row["text"] for row in rows]), len(rows)
    )
    scored = [index for index, row in enumerate(rows) if eligible(row, audit)]
    challenge_frame = pd.DataFrame(
        [{**rows[index], "true_label": rows[index]["label"]} for index in scored]
    )
    challenge = _condition(
        challenge_frame, probabilities[scored], output / "challenge", config
    )
    # The same frozen weights on the untouched original test set, not a saved
    # leaderboard mean from potentially different checkpoints/seeds.
    test = pd.read_csv(splits["test"])
    if "label" not in test or not set(test["label"]).issubset(LABELS):
        raise ValueError("Original test set must contain canonical labels")
    test["id"] = test.index.astype(str)
    test["true_label"] = test["label"]
    original_probs = validate_probabilities(
        model.probabilities(test["text"].tolist()), len(test)
    )
    original = _condition(test, original_probs, output / "original_test", config)
    predictions = [
        {
            **row,
            "scored": eligible(row, audit),
            "contamination_excluded": audit["rows"][row["id"]]["excluded"],
            "prediction": LABELS[int(probabilities[index].argmax())],
            "probabilities": dict(
                zip(LABELS, probabilities[index].tolist(), strict=True)
            ),
        }
        for index, row in enumerate(rows)
    ]
    write_json(output / "challenge_predictions.json", predictions)
    pairs = pair_changes(rows, probabilities, audit)
    write_json(output / "perturbation_changes.json", pairs)
    clean = [
        index
        for index, row in enumerate(rows)
        if not audit["rows"][row["id"]]["excluded"]
    ]
    reviewed_invariant = [
        pair for pair in pairs if pair["invariance_violation"] is not None
    ]
    summary = {
        "status": "evaluated" if scored else "awaiting_manual_labels",
        "model": model.identity,
        "comparison_note": COMPARISON_NOTE,
        "challenge_total": len(rows),
        "challenge_scored": len(scored),
        "challenge_review_status": dict(Counter(row["review_status"] for row in rows)),
        "contamination_excluded": len(rows) - len(clean),
        "robustness_macro_f1": challenge["macro_f1"] if challenge else None,
        "original_test_macro_f1_same_checkpoint": original["macro_f1"],
        "confidence_clean_candidates_unlabelled": confidence_summary(
            probabilities[clean], config["confidence_bins"]
        ),
        "reviewed_invariant_pairs": len(reviewed_invariant),
        "invariance_violation_rate": (
            (
                sum(pair["invariance_violation"] for pair in reviewed_invariant)
                / len(reviewed_invariant)
            )
            if reviewed_invariant
            else None
        ),
    }
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(
        [original_probs.max(axis=1), probabilities[clean].max(axis=1)],
        bins=np.linspace(0, 1, config["confidence_bins"] + 1),
        label=["Original test", "Clean challenge candidates (incl. unreviewed)"],
        density=False,
    )
    axis.set(
        xlabel="Maximum predicted probability (not calibrated certainty)",
        ylabel="Count",
        title="Confidence under different evaluation conditions",
    )
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output / "confidence_distribution.png", dpi=150)
    plt.close(figure)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Robustness evaluation evaluation\n\n"
        + COMPARISON_NOTE
        + "\n\n"
        + f"Reviewed clean challenge rows: {len(scored)}/{len(rows)}.\n\n"
        + f"Robustness Macro-F1: {summary['robustness_macro_f1']} "
        "(null = not scored).\n\n"
        + "See condition metrics.json for per-class precision/recall, ECE, "
        "bootstrap intervals; failure_cases.csv for categorized errors; "
        "perturbation_changes.json for paired changes. "
        "Unreviewed proposals never enter supervised scores.\n",
        encoding="utf-8",
    )
    return summary


def run(
    challenge_path: Path,
    splits: dict[str, Path],
    output: Path,
    config: dict,
    model: TextModel | None = None,
    audit_only: bool = False,
) -> dict:
    """Exclusive output directory protects all existing Phase 1 artifacts."""
    root = Path(__file__).resolve().parents[2]
    destination = output.resolve()
    for protected in (
        root / "results",
        root / "data",
        root / "src",
        root / "app",
        root / "configs",
        root / "docs",
        root / ".git",
        root / "tests",
        root / "scripts",
    ):
        if destination == protected or protected in destination.parents:
            raise ValueError(
                f"Evaluation output must be outside protected tree: {protected}"
            )
    for key in (
        "seed",
        "bootstrap_resamples",
        "confidence_bins",
        "max_explanation_words",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    configure(config["seed"])
    rows = load_challenge(challenge_path)
    audit = audit_contamination(rows, splits, config["near_duplicate_threshold"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "contamination.json", audit)
    write_json(
        output / "manifest.json",
        {
            "challenge_sha256": sha256(challenge_path),
            "reference_sha256": audit["reference_sha256"],
            "config": config,
            "python": platform.python_version(),
            "dependencies": {
                name: version(name)
                for name in ("torch", "transformers", "numpy", "pandas", "scikit-learn")
            },
            "model": model.identity if model else None,
        },
    )
    if audit_only:
        summary = {
            "status": "audit_only",
            "total": len(rows),
            "reviewed_clean": sum(eligible(row, audit) for row in rows),
            "excluded": sum(result["excluded"] for result in audit["rows"].values()),
            "robustness_macro_f1": None,
        }
        write_json(output / "summary.json", summary)
        return summary
    if model is None:
        raise ValueError("Evaluation requires an explicitly loaded frozen model")
    return evaluate(model, rows, audit, splits, output, config)
