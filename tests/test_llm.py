"""Tests for the cached OpenAI LLM benchmark."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

llm_module = pytest.importorskip(
    "models.llm",
    reason="Optional Phase 1 LLM implementation is absent from upstream; "
    "Robustness evaluation reference models do not use it.",
)
CostLatencyTracker = llm_module.CostLatencyTracker
classify_text = llm_module.classify_text


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "label": "Urgency",
                            "probs": {
                                "Urgency": 0.7,
                                "Scarcity": 0.1,
                                "Social Proof": 0.1,
                                "Normal": 0.1,
                            },
                        }
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_classification_parses_json_and_caches_without_second_call(
    tmp_path: Path,
) -> None:
    client = Mock()
    client.chat.completions.create.return_value = _response()
    tracker = CostLatencyTracker()
    first = classify_text(
        client,
        "Only two left!",
        "zero_shot",
        tmp_path,
        tracker,
        sleep_fn=lambda _: None,
    )
    second = classify_text(
        client,
        "Only two left!",
        "zero_shot",
        tmp_path,
        tracker,
        sleep_fn=lambda _: None,
    )
    assert first[0] == second[0] == "Urgency"
    assert np.isclose(first[1].sum(), 1.0)
    assert client.chat.completions.create.call_count == 1
    assert tracker.requests == 1


def test_dry_run_requires_no_api_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import pandas as pd

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.DataFrame({"text": ["100% cotton shirt"] * 4, "label": ["Normal"] * 4})
    frame.to_csv(data_dir / "train.csv", index=False)
    frame.to_csv(data_dir / "val.csv", index=False)
    frame.to_csv(data_dir / "test.csv", index=False)
    from models import llm

    monkeypatch.setattr(llm, "configured_seeds", lambda: [1729])
    llm.run(
        "zero_shot",
        data_dir=data_dir,
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        dry_run=True,
    )
    assert (tmp_path / "results/llm_gpt4o_mini/seed1729/preds.csv").is_file()
    assert (tmp_path / "results/llm_gpt4o_mini/seed1729/val_preds.csv").is_file()
