"""Human-review import must not manufacture approvals or lose provenance."""

import csv
import json

import pytest

from scripts.build_challenge import candidates
from src.robustness.review import COLUMNS, export_review, import_review


def setup_review(tmp_path):
    source = tmp_path / "candidates.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in candidates()[:4]), encoding="utf-8"
    )
    decisions = tmp_path / "review.csv"
    export_review(source, decisions)
    with decisions.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return source, decisions, rows


def save_decisions(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_blind_template_does_not_approve_proposals(tmp_path):
    source, decisions, rows = setup_review(tmp_path)
    assert all(
        row["label"] == row["reviewer"] == row["rationale"] == "" for row in rows
    )
    assert "proposed_label" not in rows[0]
    result = import_review(source, decisions, tmp_path / "reviewed.jsonl")
    assert all(
        row["review_status"] == "pending" and row["label"] is None for row in result
    )
    with pytest.raises(FileExistsError):
        export_review(source, decisions)


def test_explicit_human_decision_is_preserved(tmp_path):
    source, decisions, rows = setup_review(tmp_path)
    rows[0].update(
        review_status="approved",
        label="Urgency",
        reviewer="test-human",
        rationale="Test annotation: purchasing deadline",
    )
    save_decisions(decisions, rows)
    output = tmp_path / "reviewed.jsonl"
    reviewed = import_review(source, decisions, output)
    assert reviewed[0]["label"] == "Urgency"
    assert reviewed[0]["reviewer"] == "test-human"
    assert json.loads(source.read_text().splitlines()[0])["label"] is None


@pytest.mark.parametrize(
    "mutation", ("hash", "text", "duplicate", "missing_reviewer", "pending_label")
)
def test_invalid_review_never_creates_output(tmp_path, mutation):
    source, decisions, rows = setup_review(tmp_path)
    if mutation == "hash":
        rows[0]["dataset_sha256"] = "incorrect"
    elif mutation == "text":
        rows[0]["text"] = "changed"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing_reviewer":
        rows[0].update(review_status="approved", label="Urgency")
    else:
        rows[0]["label"] = "Urgency"
    save_decisions(decisions, rows)
    output = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError):
        import_review(source, decisions, output)
    assert not output.exists()
