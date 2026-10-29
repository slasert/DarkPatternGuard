"""Train reproducible reference models while preserving historical results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.train import train_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/cleaned")
    parser.add_argument(
        "--models-dir", type=Path, default=ROOT / "models_out/reference"
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(
        json.dumps(train_reference(args.data_dir, args.models_dir, args.seed), indent=2)
    )


if __name__ == "__main__":
    main()
