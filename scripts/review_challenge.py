"""Export a blind review CSV or import explicit human annotation decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.review import export_review, import_review, review_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import", "status"))
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/challenge/challenge.jsonl"
    )
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(review_csv(args.source), end="")
            return
        if args.output is None:
            parser.error("--output must name a new file")
        if args.action == "export":
            export_review(args.source, args.output)
        else:
            if args.decisions is None:
                parser.error("--decisions is required for import")
            import_review(args.source, args.decisions, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Review stopped: {error}\n")
    print(f"Created {args.output}; no automatic label approval was performed")


if __name__ == "__main__":
    main()
