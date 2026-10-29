"""Tests for cleaning and split artifacts."""

import json
from pathlib import Path

import pandas as pd

from data.clean import clean_records
from data.split import write_split_artifacts


def _records() -> list[dict[str, str]]:
    records = []
    for label in ("Urgency", "Scarcity", "Social Proof"):
        for index in range(8):
            records.append(
                {
                    "description": f"<b>{label} offer {index}</b> https://example.com",
                    "pattern_type": label,
                }
            )
    return records


def test_cleaning_injects_normal_and_deduplicates() -> None:
    frame = clean_records(_records() + [_records()[0]])
    assert "Normal" in set(frame["label"])
    assert frame["text"].is_unique
    assert not frame["text"].str.contains("https://", regex=False).any()


def test_splits_have_no_overlap_and_match_manifest(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.csv"
    clean_records(_records()).to_csv(cleaned, index=False)
    output = tmp_path / "artifacts"
    manifest = write_split_artifacts(cleaned, output, seed=1729)
    frames = [pd.read_csv(output / f"{name}.csv") for name in ("train", "val", "test")]
    assert all(not frame["label"].isna().any() for frame in frames)
    assert len(set(frames[0]["text"]) & set(frames[1]["text"])) == 0
    assert len(set(frames[0]["text"]) & set(frames[2]["text"])) == 0
    assert len(set(frames[1]["text"]) & set(frames[2]["text"])) == 0
    disk_manifest = json.loads((output / "splits_manifest.json").read_text())
    assert disk_manifest == manifest
    for name, frame in zip(("train", "val", "test"), frames, strict=True):
        assert manifest["splits"][name]["rows"] == len(frame)
        assert manifest["splits"][name]["class_counts"] == {
            str(label): int(count)
            for label, count in frame["label"].value_counts().items()
        }


def test_source_pages_are_kept_in_one_partition(tmp_path: Path) -> None:
    records = [
        {
            "text": f"{label} example from page {page_id}",
            "pattern_type": label,
            "source_id": str(page_id),
            "source_category": label,
        }
        for page_id in range(40)
        for label in ("Urgency", "Scarcity", "Social Proof", "Normal")
    ]
    cleaned = tmp_path / "cleaned.csv"
    clean_records(records).to_csv(cleaned, index=False)
    output = tmp_path / "grouped_artifacts"
    manifest = write_split_artifacts(cleaned, output, seed=1729)
    frames = [pd.read_csv(output / f"{name}.csv") for name in ("train", "val", "test")]
    source_ids = [set(frame["source_id"]) for frame in frames]

    assert manifest["grouped_by_source_id"] is True
    assert not source_ids[0] & source_ids[1]
    assert not source_ids[0] & source_ids[2]
    assert not source_ids[1] & source_ids[2]
