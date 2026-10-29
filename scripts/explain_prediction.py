"""Export actual frozen-model explanations as JSON and presentation-ready HTML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.evaluate import configure  # noqa: E402
from src.robustness.explain import explain  # noqa: E402
from src.robustness.inference import ArtifactUnavailable, load_checkpoint  # noqa: E402
from src.robustness.visualize import attribution_html  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "finetune_bert",
            "finetune_deberta",
            "reference_logistic_regression",
            "reference_naive_bayes",
        ),
    )
    parser.add_argument("--checkpoint-seed", type=int, default=1729)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models_out")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/robustness.json")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    args = parser.parse_args()
    try:
        destination = args.output.resolve()
        for protected in (ROOT / "results", ROOT / "data"):
            if destination == protected or protected in destination.parents:
                raise ValueError(
                    "Explanation outputs must be separate from Phase 1 data/results"
                )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        configure(config["seed"])
        model = load_checkpoint(args.model, args.checkpoint_seed, args.models_dir)
        result = explain(model, args.text, max_words=config["max_explanation_words"])
        rendered = attribution_html(result)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "explanation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (args.output / "explanation.html").write_text(rendered, encoding="utf-8")
    except (ArtifactUnavailable, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Explanation stopped: {error}\n")
    print(f"Actual-model explanation written to {args.output}")


if __name__ == "__main__":
    main()
