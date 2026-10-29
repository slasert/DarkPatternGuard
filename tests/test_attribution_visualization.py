"""A presentation export must retain actual scores and safely render text."""

from src.robustness.visualize import attribution_html


def test_html_preserves_signed_values_identity_and_escapes_text():
    result = {
        "text": "<script> sale",
        "label": "Urgency",
        "confidence": 0.7,
        "model": {"name": "fixture", "checkpoint_sha256": "fixture-only"},
        "description": "Sensitivity, not causal evidence.",
        "truncation_warning": "256 subword limit",
        "tokens": [{"text": "sale", "start": 9, "end": 13, "score": -0.12}],
    }
    document = attribution_html(result)
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert "-12.000 percentage points" in document
    assert "70.00% model confidence" in document
    assert "fixture-only" in document
    assert "not causal evidence" in document
