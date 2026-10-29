"""Actual fitting/JSON roundtrip, no test/challenge fitting, and API integration."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.models_registry import ModelRegistry, set_registry
from src.eval.metrics import LABELS
from src.robustness.inference import ArtifactUnavailable, from_registry, load_checkpoint
from src.robustness.perturb import run_perturbations
from src.robustness.reference import ReferenceModel
from src.robustness.train import train_reference


@pytest.fixture
def fitted(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    texts = [
        "hurry deadline tonight",
        "only two units remain",
        "buyers recommend popular",
        "ceramic kettle handle",
    ]
    frame = pd.DataFrame({"text": texts * 3, "label": list(LABELS) * 3})
    for split in ("train", "val", "test"):
        frame.to_csv(data / f"{split}.csv", index=False)
    root = tmp_path / "models"
    train_reference(data, root / "reference")
    return data, root


def test_json_models_probabilities_and_exact_logit_contributions(fitted):
    _, root = fitted
    for name in ("reference_logistic_regression", "reference_naive_bayes"):
        model = load_checkpoint(name, 1729, root)
        probabilities = model.probabilities(["hurry deadline tonight"])
        assert probabilities.shape == (1, 4)
        assert probabilities.sum() == pytest.approx(1)
        contributions = model.feature_contributions("hurry deadline tonight")
        assert sum(item["score"] for item in contributions["features"]) + contributions[
            "bias"
        ] == pytest.approx(contributions["logit"])
        assert not model.identity["training"]["challenge_used_for_training"]
        assert not model.identity["training"]["test_used_for_selection"]
        path = model.path
        payload = json.loads(path.read_text())
        payload["weights"][0][0] = float("nan")
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactUnavailable, match="Invalid"):
            ReferenceModel(path)


def test_registry_and_real_api_explain_use_same_reference(fitted):
    _, root = fitted
    registry = ModelRegistry(models_dir=root)
    registry.load()
    model = from_registry(registry)
    set_registry(registry)
    try:
        with TestClient(app) as client:
            text = "hurry deadline tonight"
            prediction = client.post("/predict", json={"text": text})
            explanation = client.post("/explain", json={"text": text}).json()
            assert prediction.status_code == 200
            assert prediction.json()["label"] == explanation["label"]
            assert explanation["confidence"] == pytest.approx(
                float(model.probabilities([text]).max())
            )
            assert "feature_attributions" in explanation
            assert explanation["model"]["training_provenance"].startswith(
                "new authorized"
            )
    finally:
        set_registry(None)


def test_reference_training_is_deterministic_without_reading_test(fitted, tmp_path):
    data, root = fitted
    (data / "test.csv").unlink()
    train_reference(data, tmp_path / "repeat")
    for name in ("reference_logistic_regression", "reference_naive_bayes"):
        first = root / "reference" / name / "seed1729.json"
        second = tmp_path / "repeat" / name / "seed1729.json"
        assert first.read_bytes() == second.read_bytes()


def test_metamorphic_metrics_require_explicit_label_invariance(fitted, tmp_path):
    data, root = fitted
    model = load_checkpoint("reference_logistic_regression", 1729, root)
    summary = run_perturbations(model, data / "test.csv", tmp_path / "probe")
    assert summary["conditions"]["uppercase"]["prediction_flips"] == 0
    assert summary["conditions"]["expanded_whitespace"]["prediction_flips"] == 0
    assert (
        "macro_f1_inherited_labels" not in summary["conditions"]["single_internal_typo"]
    )
    np.testing.assert_allclose(
        model.probabilities(["hurry"]), model.probabilities(["HURRY"])
    )
