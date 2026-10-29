"""Challenge schema validation, manual-label gates and contamination audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.clean import clean_text
from src.eval.metrics import LABELS
from src.robustness.inference import sha256

CATEGORIES = (
    "paraphrase",
    "negation",
    "ambiguous_promotion",
    "mixed_signals",
    "neutral_keywords",
    "typo_punctuation",
)


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", clean_text(text)))


def load_challenge(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate_rows(rows)


def validate_rows(rows: list[dict]) -> list[dict]:
    """Validate candidate or explicitly reviewed records without writing files."""
    if not rows:
        raise ValueError("Challenge dataset is empty")
    identifiers = set()
    for row in rows:
        required = {
            "id",
            "text",
            "proposed_label",
            "label",
            "review_status",
            "reviewer",
            "rationale",
            "category",
            "source",
            "parent_id",
            "relation",
        }
        if required - row.keys():
            raise ValueError(f"Challenge row missing fields: {required - row.keys()}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in identifiers:
            raise ValueError("Challenge IDs must be unique nonempty strings")
        identifiers.add(row["id"])
        if (
            not isinstance(row["text"], str)
            or not normalized(row["text"])
            or len(row["text"].split()) > 500
        ):
            raise ValueError(f"Invalid challenge text: {row['id']}")
        if row["category"] not in CATEGORIES or row["proposed_label"] not in LABELS:
            raise ValueError(f"Invalid category/proposed label: {row['id']}")
        if row["review_status"] not in ("pending", "approved", "ambiguous", "rejected"):
            raise ValueError(f"Invalid review status: {row['id']}")
        if row["label"] is not None and row["label"] not in LABELS:
            raise ValueError(f"Invalid reviewed label: {row['id']}")
        if row["review_status"] == "approved" and (
            row["label"] not in LABELS
            or not isinstance(row["reviewer"], str)
            or not row["reviewer"].strip()
            or not isinstance(row["rationale"], str)
            or not row["rationale"].strip()
        ):
            raise ValueError(
                f"Approved label needs reviewer and rationale: {row['id']}"
            )
        if row["relation"] not in (None, "invariant", "label_change", "uncertain"):
            raise ValueError(f"Invalid perturbation relation: {row['id']}")
        if bool(row["parent_id"]) != bool(row["relation"]):
            raise ValueError(f"Pair needs parent and relation together: {row['id']}")
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        parent = row["parent_id"]
        if parent is not None and (parent not in by_id or parent == row["id"]):
            raise ValueError(f"Unknown/self parent: {row['id']}")
        if parent and by_id[parent]["parent_id"]:
            raise ValueError(
                "Perturbations must refer to an original, not another variant"
            )
        if (
            parent
            and row["review_status"] == by_id[parent]["review_status"] == "approved"
        ):
            same = row["label"] == by_id[parent]["label"]
            if row["relation"] == "invariant" and not same:
                raise ValueError("Invariant pair has conflicting approved labels")
            if row["relation"] == "label_change" and same:
                raise ValueError("Label-change pair has identical approved labels")
    return rows


def audit_contamination(
    rows: list[dict], splits: dict[str, Path], threshold: float
) -> dict:
    """Compare to all Phase 1 splits, never fit a classifier on challenge text.

    TF-IDF here is a deterministic similarity detector, not a training feature
    pipeline. Report every match >= threshold, not only the nearest neighbour.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Near-duplicate threshold must be in (0, 1]")
    references, provenance = [], []
    hashes = {}
    for split, path in splits.items():
        frame = pd.read_csv(path)
        if "text" not in frame or frame.empty or frame["text"].isna().any():
            raise ValueError(f"Missing/empty reference text in {path}")
        hashes[split] = sha256(path)
        for index, text in enumerate(frame["text"]):
            references.append(normalized(str(text)))
            provenance.append({"split": split, "row": index})
    challenge = [normalized(row["text"]) for row in rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(references + challenge)
    similarities = cosine_similarity(
        matrix[len(references) :], matrix[: len(references)]
    )
    internal = cosine_similarity(matrix[len(references) :])
    results = {}
    for index, row in enumerate(rows):
        matches = []
        for ref_index, reference in enumerate(references):
            exact = challenge[index] == reference
            score = float(similarities[index, ref_index])
            if exact or score >= threshold:
                matches.append(
                    {
                        **provenance[ref_index],
                        "similarity": score,
                        "kind": "exact" if exact else "near",
                    }
                )
        related = []
        for other in range(index):
            is_pair = (
                row["parent_id"] == rows[other]["id"]
                or rows[other]["parent_id"] == row["id"]
                or (
                    row["parent_id"] is not None
                    and row["parent_id"] == rows[other]["parent_id"]
                )
            )
            if (
                challenge[index] == challenge[other]
                or internal[index, other] >= threshold
            ):
                related.append(
                    {
                        "id": rows[other]["id"],
                        "declared_pair": is_pair,
                        "similarity": float(internal[index, other]),
                        "exact": clean_text(row["text"])
                        == clean_text(rows[other]["text"]),
                    }
                )
        results[row["id"]] = {
            "excluded": bool(matches),
            "reference_matches": matches,
            "within_challenge_matches": related,
        }
    # Quarantine both members of undeclared near/exact duplicates and all exact
    # duplicates (even declared pairs); retain intended nonidentical paired edits.
    for result in results.values():
        for match in result["within_challenge_matches"]:
            if match["exact"] or not match["declared_pair"]:
                result["excluded"] = True
                results[match["id"]]["excluded"] = True
    return {
        "threshold": threshold,
        "method": "NFKC/HTML/lower/alphanumeric exact + "
        "char_wb TF-IDF(3,5) cosine; lexical heuristic, not semantic guarantee",
        "reference_sha256": hashes,
        "rows": results,
    }


def eligible(row: dict, audit: dict) -> bool:
    return (
        row["review_status"] == "approved" and not audit["rows"][row["id"]]["excluded"]
    )
