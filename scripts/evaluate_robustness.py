"""Audit challenge data or evaluate frozen models without changing training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.robustness.evaluate import run  # noqa: E402
from src.robustness.inference import ArtifactUnavailable, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenge", type=Path, default=ROOT / "data/challenge/challenge.jsonl"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robustness.json")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/cleaned")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models_out")
    parser.add_argument(
        "--model",
        choices=(
            "finetune_bert",
            "finetune_deberta",
            "reference_logistic_regression",
            "reference_naive_bayes",
        ),
    )
    parser.add_argument("--checkpoint-seed", type=int, default=1729)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside results/"
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and not args.model:
        parser.error(
            "--model is required for evaluation; use --audit-only without weights"
        )
    try:
        model = (
            None
            if args.audit_only
            else load_checkpoint(args.model, args.checkpoint_seed, args.models_dir)
        )
        summary = run(
            args.challenge,
            {
                split: args.splits_dir / f"{split}.csv"
                for split in ("train", "val", "test")
            },
            args.output,
            json.loads(args.config.read_text(encoding="utf-8")),
            model=model,
            audit_only=args.audit_only,
        )
    except (ArtifactUnavailable, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Robustness evaluation stopped: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
