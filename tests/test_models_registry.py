from __future__ import annotations

import numpy as np
import pandas as pd

from app.backend.models_registry import (
    LABELS,
    PROBABILITY_COLUMNS,
    ModelRegistry,
)


def _write_validation_file(path, predicted_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.full((len(LABELS), len(LABELS)), 0.01)
    for row, label in enumerate(predicted_labels):
        probabilities[row, LABELS.index(label)] = 0.97
    pd.DataFrame(
        {
            "true_label": LABELS,
            **{
                column: probabilities[:, index]
                for index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    ).to_csv(path, index=False)


def test_registry_selects_best_transformer_using_validation_scores(tmp_path) -> None:
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    for model_name in ("finetune_bert", "finetune_deberta"):
        (models_dir / model_name).mkdir(parents=True)
        (models_dir / model_name / "seed1729.weights.pt").touch()
    _write_validation_file(
        results_dir / "finetune_bert" / "seed1729" / "val_preds.csv",
        ["Urgency", "Scarcity", "Social Proof", "Urgency"],
    )
    _write_validation_file(
        results_dir / "finetune_deberta" / "seed1729" / "val_preds.csv",
        list(LABELS),
    )

    registry = ModelRegistry(results_dir=results_dir, models_dir=models_dir)

    assert registry._best_transformer_candidate() == ("finetune_deberta", 1729)


def test_leaderboard_metadata_marks_only_the_loaded_model_active(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {"model": "finetune_bert", "mean_macro_f1": 0.7},
            {"model": "finetune_deberta", "mean_macro_f1": 0.8},
        ]
    ).to_csv(results_dir / "leaderboard.csv", index=False)
    registry = ModelRegistry(results_dir=results_dir)
    registry._models = {"finetune_deberta": object()}
    registry._loaded = True

    metadata = registry.model_metadata()

    assert [entry["active"] for entry in metadata] == [False, True]
    assert [entry["version"] for entry in metadata] == ["trained", "trained"]


def test_registry_selects_among_restored_checkpoints(tmp_path):
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    for seed in (1729, 2718):
        _write_validation_file(
            results_dir / "finetune_bert" / f"seed{seed}" / "val_preds.csv",
            list(LABELS),
        )
    checkpoint = models_dir / "finetune_bert" / "seed2718.weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    registry = ModelRegistry(results_dir=results_dir, models_dir=models_dir)
    assert registry._best_transformer_candidate() == ("finetune_bert", 2718)
