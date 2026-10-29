"""Smoke-check real reference artifacts, probabilities and model explanations."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.config import configured_seeds  # noqa: E402
from src.robustness.explain import explain  # noqa: E402
from src.robustness.reference import REFERENCE_MODELS, ReferenceModel  # noqa: E402
from src.robustness.train import train_reference  # noqa: E402


def check_models(models_dir: Path, seed: int) -> None:
    """Check actual loaded weights, not synthetic hard-coded predictions."""
    text = "Only two units remain in stock."
    for name in REFERENCE_MODELS:
        model = ReferenceModel(models_dir / name / f"seed{seed}.json")
        result = explain(model, text)
        assert (
            result["model"]["checkpoint_sha256"] == model.identity["checkpoint_sha256"]
        )
        assert result["confidence"] == float(model.probabilities([text]).max())
        assert result["feature_attributions"]["features"]
    print("[DarkPatternGuard] smoke: trained reference models and explanations passed")


def main() -> None:
    """Use existing weights, or exercise training in a disposable directory."""
    seed = configured_seeds(ROOT / "configs/base.yaml")[0]
    models_dir = ROOT / "models_out/reference"
    if all(
        (models_dir / name / f"seed{seed}.json").is_file() for name in REFERENCE_MODELS
    ):
        check_models(models_dir, seed)
    else:
        with TemporaryDirectory(prefix="darkpatternguard-smoke-") as temporary:
            temporary_models = Path(temporary) / "models"
            train_reference(ROOT / "data/cleaned", temporary_models, seed)
            check_models(temporary_models, seed)


if __name__ == "__main__":
    main()
