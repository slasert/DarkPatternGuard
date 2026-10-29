"""Fetch the published Princeton-derived e-commerce text dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

SOURCE_COMMIT = "85ea9df31928d5ad6ea0b106316b441c201adbbe"
SOURCE_URL = (
    "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/"
    f"{SOURCE_COMMIT}/dataset/dataset.tsv"
)
SOURCE_REPOSITORY = "https://github.com/yamanalab/ec-darkpattern"
LABEL_MAP = {
    "Urgency": "Urgency",
    "Scarcity": "Scarcity",
    "Social Proof": "Social Proof",
    "Not Dark Pattern": "Normal",
}
EXPECTED_COLUMNS = {"page_id", "text", "label", "Pattern Category"}


def records_from_tsv(contents: str) -> tuple[list[dict[str, str]], Counter[str]]:
    """Retain the three requested pattern categories and real non-pattern text."""
    reader = csv.DictReader(io.StringIO(contents), delimiter="\t")
    if reader.fieldnames is None or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
        raise ValueError(
            "Upstream TSV schema changed; expected columns: "
            + ", ".join(sorted(EXPECTED_COLUMNS))
        )

    records: list[dict[str, str]] = []
    excluded_categories: Counter[str] = Counter()
    for row in reader:
        category = row["Pattern Category"].strip()
        label = LABEL_MAP.get(category)
        if label is None:
            excluded_categories[category] += 1
            continue
        text = row["text"]
        if not text.strip():
            excluded_categories["empty_text"] += 1
            continue
        expected_binary_label = "0" if label == "Normal" else "1"
        if row["label"].strip() != expected_binary_label:
            raise ValueError(
                f"Upstream binary label conflicts with category {category!r}"
            )
        records.append(
            {
                "text": text,
                "pattern_type": label,
                "source_id": row["page_id"].strip(),
                "source_category": category,
                "source_label": row["label"].strip(),
                "source_dataset": "yamanalab/ec-darkpattern dataset.tsv",
                "source_url": SOURCE_URL,
            }
        )

    if not 1500 <= len(records) <= 2500:
        raise ValueError(
            f"Expected 1,500–2,500 selected source records; got {len(records)}"
        )
    counts = Counter(record["pattern_type"] for record in records)
    missing_labels = set(LABEL_MAP.values()) - counts.keys()
    if missing_labels:
        raise ValueError(
            "Upstream dataset lacks required labels: " f"{sorted(missing_labels)}"
        )
    return records, excluded_categories


def fetch_raw_dataset(output_path: Path) -> dict[str, object]:
    """Download and save a provenance-preserving raw JSON array."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw data: {output_path}")
    with urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    contents = payload.decode("utf-8-sig")
    records, excluded_categories = records_from_tsv(contents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_rows": len(records),
        "class_counts": dict(Counter(record["pattern_type"] for record in records)),
        "excluded_category_counts": dict(excluded_categories),
        "output_path": str(output_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/princeton_dark_patterns_raw.json"),
    )
    args = parser.parse_args()
    print(json.dumps(fetch_raw_dataset(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
