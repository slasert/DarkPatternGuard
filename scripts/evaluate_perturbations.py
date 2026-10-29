"""Run reproducible, held-out metamorphic probes using a frozen reference model."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.evaluate import configure  # noqa: E402
from src.robustness.inference import load_checkpoint  # noqa: E402
from src.robustness.perturb import run_perturbations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args.seed)
    model = load_checkpoint(args.model, args.seed, ROOT / "models_out")
    print(
        json.dumps(
            run_perturbations(
                model, ROOT / "data/cleaned/test.csv", args.output, args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
