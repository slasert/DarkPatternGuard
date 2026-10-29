"""Robustness correctness tests. Fixture models are never research results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.backend.main as api
from app.backend.models_registry import ModelRegistry, set_registry
from scripts.build_challenge import candidates
from src.eval.metrics import LABELS
from src.robustness.challenge import audit_contamination, eligible, load_challenge
from src.robustness.evaluate import pair_changes, run
from src.robustness.explain import explain
from src.robustness.inference import (
    ArtifactUnavailable,
    from_registry,
    load_checkpoint,
    validate_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "seed": 1729,
    "near_duplicate_threshold": 0.85,
    "bootstrap_resamples": 10,
    "confidence_bins": 10,
    "max_explanation_words": 80,
}


class FixtureModel:
    """Deterministic probability oracle for assertions only, not a trained model."""

    identity = {
        "name": "test_fixture",
        "seed": 1729,
        "checkpoint_sha256": "fixture-only",
        "tokenizer_sha256": "fixture-only",
        "labels": list(LABELS),
    }

    def probabilities(self, texts):
        probabilities = []
        for text in texts:
            scores = np.array([1.0, 1.0, 1.0, 2.0])
            for index, word in enumerate(("midnight", "warehouse", "shoppers")):
                if word in text.lower():
                    scores[index] += 8
            probabilities.append(scores / scores.sum())
        return np.asarray(probabilities)


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _splits(tmp_path):
    paths = {}
    for split in ("train", "val", "test"):
        paths[split] = tmp_path / f"{split}.csv"
        pd.DataFrame(
            {
                "text": [
                    f"{split} {label} completely unrelated reference {i}"
                    for i, label in enumerate(LABELS)
                ],
                "label": LABELS,
            }
        ).to_csv(paths[split], index=False)
    return paths


def _clean_audit(rows):
    return {"rows": {row["id"]: {"excluded": False} for row in rows}}


def test_candidates_are_reproducible_separate_and_unapproved():
    rows = load_challenge(ROOT / "data/challenge/challenge.jsonl")
    assert rows == candidates()
    assert len(rows) == 36
    assert all(row["label"] is None and row["reviewer"] is None for row in rows)
    assert {row["proposed_label"] for row in rows} == set(LABELS)
    assert len({row["category"] for row in rows}) == 6
    assert sum(row["review_status"] == "ambiguous" for row in rows) == 8
    assert not any(eligible(row, _clean_audit(rows)) for row in rows)


def test_manual_approval_requires_label_reviewer_rationale(tmp_path):
    row = {**candidates()[0], "review_status": "approved", "label": "Urgency"}
    path = tmp_path / "challenge.jsonl"
    _write_rows(path, [row])
    with pytest.raises(ValueError, match="reviewer"):
        load_challenge(path)
    row["reviewer"] = "test-annotator"
    _write_rows(path, [row])
    reviewed = load_challenge(path)
    assert eligible(reviewed[0], _clean_audit(reviewed))
    reviewed[0]["review_status"] = "ambiguous"
    assert not eligible(reviewed[0], _clean_audit(reviewed))


def test_duplicate_ids_and_pair_label_conflicts_are_rejected(tmp_path):
    path = tmp_path / "challenge.jsonl"
    _write_rows(path, [candidates()[0]] * 2)
    with pytest.raises(ValueError, match="unique"):
        load_challenge(path)
    rows = [candidates()[0], candidates()[24]]
    for index, row in enumerate(rows):
        row.update(review_status="approved", reviewer="fixture", label=LABELS[index])
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="Invariant"):
        load_challenge(path)


def test_contamination_checks_every_split_and_near_duplicates(tmp_path):
    paths = _splits(tmp_path)
    rows = candidates()[:3]
    rows[0]["text"] = "<b>TRAIN Urgency completely unrelated reference 0!!!</b>"
    rows[1]["text"] = "val Scarcity completely unrelated refernce 1"
    rows[2]["text"] = "test Social Proof completely unrelated reference 2"
    audit = audit_contamination(rows, paths, 0.65)
    assert all(value["excluded"] for value in audit["rows"].values())
    assert audit["rows"][rows[0]["id"]]["reference_matches"][0]["kind"] == "exact"
    assert any(
        match["kind"] == "near"
        for match in audit["rows"][rows[1]["id"]]["reference_matches"]
    )
    assert set(audit["reference_sha256"]) == {"train", "val", "test"}


def test_declared_surface_pairs_retained_but_undeclared_duplicates_quarantined(
    tmp_path,
):
    rows = [candidates()[0], candidates()[24]]
    audit = audit_contamination(rows, _splits(tmp_path), 0.85)
    assert not any(result["excluded"] for result in audit["rows"].values())
    rows[1].update(parent_id=None, relation=None)
    audit = audit_contamination(rows, _splits(tmp_path), 0.85)
    assert all(result["excluded"] for result in audit["rows"].values())


@pytest.mark.parametrize(
    "values",
    [
        [[np.nan, 0, 0, 1]],
        [[-1, 1, 0, 1]],
        [[0.1, 0.1, 0.1, 0.1]],
        [[1, 0, 0]],
    ],
)
def test_invalid_model_outputs_are_rejected(values):
    with pytest.raises(ValueError, match="distributions"):
        validate_probabilities(values, 1)


def test_attribution_is_exact_probability_change_for_fixed_target_and_offsets():
    model = FixtureModel()
    text = "🛒 midnight, midnight warehouse!"
    result = explain(model, text)
    target = LABELS.index(result["label"])
    for token in result["tokens"]:
        assert text[token["start"] : token["end"]] == token["text"]
        deleted = text[: token["start"]] + text[token["end"] :]
        expected = (
            model.probabilities([text])[0, target]
            - model.probabilities([deleted])[0, target]
        )
        assert token["score"] == pytest.approx(expected)
    assert result["model"] == model.identity
    assert result["target_class"] == result["label"]
    assert "not causal evidence" in result["description"]
    assert result["forward_evaluations"] == len(result["tokens"]) + 1


def test_influential_and_opposing_scores_and_length_limit():
    result = explain(FixtureModel(), "midnight warehouse")
    assert any(token["score"] > 0 for token in result["tokens"])
    assert any(token["score"] < 0 for token in result["tokens"])
    with pytest.raises(ValueError, match="at most"):
        explain(FixtureModel(), "word " * 81)
    with pytest.raises(ValueError, match="at least one"):
        explain(FixtureModel(), "!!!")


def test_prediction_flips_are_not_automatically_failures():
    rows = [candidates()[0], candidates()[-4]]
    probabilities = FixtureModel().probabilities([row["text"] for row in rows])
    changes = pair_changes(rows, probabilities, _clean_audit(rows))
    assert changes[0]["relation"] == "label_change"
    assert changes[0]["invariance_violation"] is None
    assert changes[0]["both_predictions_correct"] is None


def test_fallback_and_missing_checkpoint_are_refused(tmp_path):
    registry = ModelRegistry(
        results_dir=tmp_path / "empty", models_dir=tmp_path / "empty-models"
    )
    with pytest.raises(ArtifactUnavailable, match="trained model|trained transformer"):
        from_registry(registry)
    with pytest.raises(ArtifactUnavailable, match="Restore original"):
        load_checkpoint("finetune_bert", 1729, tmp_path)
    with pytest.raises(ArtifactUnavailable, match="Unsupported"):
        load_checkpoint("lightgbm", 1729, tmp_path)


def test_tiny_transformer_checkpoint_roundtrip_and_identity(tmp_path, monkeypatch):
    """Real Torch/HF loading path, tiny random fixture, entirely offline."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer,
    )

    config = BertConfig(
        vocab_size=9,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=4,
    )
    torch.manual_seed(1729)
    original = BertForSequenceClassification(config).eval()
    artifact_dir = tmp_path / "finetune_bert"
    artifact_dir.mkdir()
    torch.save(original.state_dict(), artifact_dir / "seed1729.weights.pt")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nmidnight\nwarehouse\na\n.\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: BertForSequenceClassification(config),
    )
    model = load_checkpoint("finetune_bert", 1729, tmp_path)
    text = "midnight warehouse"
    with torch.inference_mode():
        expected = (
            original(**tokenizer(text, return_tensors="pt")).logits.softmax(1).numpy()
        )
    np.testing.assert_allclose(model.probabilities([text]), expected, atol=1e-7)
    assert len(model.identity["checkpoint_sha256"]) == 64
    assert len(model.identity["tokenizer_sha256"]) == 64
    result = explain(model, text)
    assert result["confidence"] == pytest.approx(float(expected.max()))
    torch.save({"bad_key": torch.zeros(1)}, artifact_dir / "seed1729.weights.pt")
    with pytest.raises(ArtifactUnavailable, match="Cannot load"):
        load_checkpoint("finetune_bert", 1729, tmp_path)
    (artifact_dir / "seed1729.weights.pt").write_bytes(b"invalid checkpoint bytes")
    with pytest.raises(ArtifactUnavailable, match="Invalid trained checkpoint"):
        load_checkpoint("finetune_bert", 1729, tmp_path)


def test_pipeline_only_scores_reviewed_clean_labels_and_preserves_inputs(tmp_path):
    paths = _splits(tmp_path)
    rows = candidates()[:4]
    for row in rows:
        row.update(
            review_status="approved", label=row["proposed_label"], reviewer="fixture"
        )
    rows[0]["review_status"] = "pending"
    path = tmp_path / "challenge.jsonl"
    _write_rows(path, rows)
    before = {p: p.read_bytes() for p in [*paths.values(), path]}
    output = tmp_path / "run"
    summary = run(path, paths, output, CONFIG, FixtureModel())
    assert summary["challenge_scored"] == 3
    assert summary["robustness_macro_f1"] is not None
    metrics = json.loads((output / "challenge/metrics.json").read_text())
    assert sum(row["support"] for row in metrics["per_class"].values()) == 3
    assert (output / "challenge/confusion_matrix.png").exists()
    assert (output / "original_test/confusion_matrix.png").exists()
    assert (output / "confidence_distribution.png").exists()
    assert all(p.read_bytes() == content for p, content in before.items())
    second = run(path, paths, tmp_path / "run2", CONFIG, FixtureModel())
    assert summary == second
    assert (output / "challenge/metrics.json").read_bytes() == (
        tmp_path / "run2/challenge/metrics.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        run(path, paths, output, CONFIG, FixtureModel())


def test_unreviewed_pipeline_has_null_scores_and_protected_output(tmp_path):
    paths = _splits(tmp_path)
    path = tmp_path / "challenge.jsonl"
    _write_rows(path, candidates()[:4])
    summary = run(path, paths, tmp_path / "pending", CONFIG, FixtureModel())
    assert summary["status"] == "awaiting_manual_labels"
    assert summary["robustness_macro_f1"] is None
    assert not (tmp_path / "pending/challenge").exists()
    with pytest.raises(ValueError, match="protected"):
        run(path, paths, ROOT / "results/robustness", CONFIG, audit_only=True)


def test_explanation_endpoint_comparison_and_validation(monkeypatch):
    registry = ModelRegistry()
    registry._loaded = True
    set_registry(registry)
    monkeypatch.setattr(api, "from_registry", lambda *a: FixtureModel())
    try:
        with TestClient(api.app) as client:
            response = client.post(
                "/explain",
                json={"text": "midnight sale", "perturbed_text": "plain item"},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["label"] == "Urgency"
            assert payload["comparison"]["label"] == "Normal"
            assert payload["comparison"]["prediction_changed"]
            assert payload["model"]["name"] == "test_fixture"
            assert (
                client.post("/explain", json={"text": "word " * 81}).status_code == 422
            )
            assert (
                client.post(
                    "/explain", json={"text": "abc", "perturbed_text": " "}
                ).status_code
                == 422
            )
    finally:
        set_registry(None)


def test_missing_weights_allow_actionable_api_errors(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "leaderboard.csv").write_text("model,mean_macro_f1\n", encoding="utf-8")
    set_registry(ModelRegistry(results_dir=results, models_dir=tmp_path / "missing"))
    try:
        with TestClient(api.app) as client:
            assert client.get("/docs").status_code == 200
            response = client.post("/explain", json={"text": "midnight sale"})
            assert response.status_code == 503
            assert "weights" in response.json()["detail"]
            assert client.post("/predict", json={"text": "abc"}).status_code == 503
            assert client.get("/health").status_code == 503
    finally:
        set_registry(None)
