"""Blind human-review exchange: no synthetic proposal becomes a gold label."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from src.robustness.challenge import load_challenge, validate_rows
from src.robustness.inference import sha256

COLUMNS = (
    "dataset_sha256",
    "id",
    "text",
    "category",
    "parent_id",
    "relation",
    "review_status",
    "label",
    "reviewer",
    "rationale",
)


def export_review(source: Path, output: Path) -> None:
    rows = load_challenge(source)
    digest = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            # Hide proposed labels/reasons to reduce anchoring. No model outputs
            # are included, and no synthetic proposal is prefilled as approval.
            writer.writerow(
                {
                    "dataset_sha256": digest,
                    "id": row["id"],
                    "text": row["text"],
                    "category": row["category"],
                    "parent_id": row["parent_id"],
                    "relation": row["relation"],
                    "review_status": "pending",
                    "label": "",
                    "reviewer": "",
                    "rationale": "",
                }
            )


def import_review(source: Path, decisions: Path, output: Path) -> list[dict]:
    """Validate an explicit, complete review file before creating any output."""
    rows = load_challenge(source)
    digest = sha256(source)
    with decisions.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(COLUMNS):
            raise ValueError("Review CSV columns do not match the exported template")
        decisions_list = list(reader)
    if any(
        set(row) != set(COLUMNS) or any(value is None for value in row.values())
        for row in decisions_list
    ):
        raise ValueError("Review CSV has incomplete rows or extra cells")
    by_id = {row["id"]: row for row in decisions_list}
    if len(by_id) != len(decisions_list) or set(by_id) != {r["id"] for r in rows}:
        raise ValueError(
            "Review IDs must match the full source exactly, without duplicates"
        )
    reviewed = []
    for row in rows:
        decision = by_id[row["id"]]
        if decision["dataset_sha256"] != digest:
            raise ValueError("Review belongs to a different dataset version")
        for key in ("text", "category", "parent_id", "relation"):
            if decision[key] != (row[key] or ""):
                raise ValueError(f"Review changed source field {key}: {row['id']}")
        status = decision["review_status"].strip()
        label = decision["label"].strip() or None
        reviewer = decision["reviewer"].strip() or None
        rationale = decision["rationale"].strip()
        if status != "pending" and (not reviewer or not rationale):
            raise ValueError(f"A decision needs a reviewer and rationale: {row['id']}")
        if status != "approved" and label is not None:
            raise ValueError(f"Unapproved rows must have an empty label: {row['id']}")
        reviewed.append(
            {
                **row,
                "review_status": status,
                "label": label,
                "reviewer": reviewer,
                "rationale": rationale,
            }
        )
    validate_rows(reviewed)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviewed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return reviewed


def review_csv(source: Path) -> str:
    """Return a read-only coverage summary for annotation handoff."""
    rows = load_challenge(source)
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["status", "count"])
    for status in ("pending", "ambiguous", "approved", "rejected"):
        writer.writerow([status, sum(row["review_status"] == status for row in rows)])
    return stream.getvalue()
