"""Word-span deletion attribution for any validated probability classifier."""

from __future__ import annotations

import re

from src.eval.metrics import LABELS
from src.robustness.inference import TextModel, validate_probabilities

LIMITATIONS = (
    "Each score is P(original predicted class | full text) minus P(the same "
    "class | text with this word removed). Positive supports the prediction; "
    "negative opposes it. These are model sensitivities, not causal evidence, "
    "SHAP values or additive contributions. Deletion may change grammar, "
    "tokenization and context; interactions are not separated."
)


def explain(model: TextModel, text: str, max_words: int = 80) -> dict:
    """Explain one model's own prediction, keeping target class fixed."""
    spans = list(re.finditer(r"\b\w+(?:['’-]\w+)*\b", text, flags=re.UNICODE))
    if not text.strip() or not spans:
        raise ValueError("Explanation needs at least one word")
    if len(spans) > max_words:
        raise ValueError(f"Explanation supports at most {max_words} word spans")
    original = validate_probabilities(model.probabilities([text]), 1)[0]
    target = int(original.argmax())
    variants = [text[: span.start()] + text[span.end() :] for span in spans]
    perturbed = validate_probabilities(model.probabilities(variants), len(variants))
    result = {
        "text": text,
        "model": model.identity,
        "label": LABELS[target],
        "confidence": float(original[target]),
        "probabilities": dict(zip(LABELS, original.tolist(), strict=True)),
        "method": "word_deletion_probability_difference",
        "target_class": LABELS[target],
        "baseline": "remove one word span, retain surrounding characters",
        "description": LIMITATIONS,
        "tokens": [
            {
                "text": span.group(),
                "start": span.start(),
                "end": span.end(),
                "score": float(original[target] - perturbed[index, target]),
            }
            for index, span in enumerate(spans)
        ],
        "forward_evaluations": 1 + len(spans),
        "max_model_tokens": model.identity.get("max_length", 256),
        "truncation_warning": (
            "Serving truncates to 256 subword tokens. Deletion can move words into "
            "the visible context; short texts are recommended."
        ),
    }
    if hasattr(model, "feature_contributions"):
        result["feature_attributions"] = model.feature_contributions(text)
        result["truncation_warning"] = (
            "This TF-IDF model uses the full text without subword truncation. "
            "Deletion renormalizes TF-IDF and changes overlapping n-grams."
        )
    return result
