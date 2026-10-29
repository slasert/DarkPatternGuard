"""Read evaluation seeds without importing unavailable training modules."""

from pathlib import Path

import yaml


def configured_seeds(config_path: Path = Path("configs/base.yaml")) -> list[int]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [int(seed) for seed in payload.get("seeds", [payload["seed"]])]
