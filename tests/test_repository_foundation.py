"""Checks for the repository foundation."""

from pathlib import Path


def test_required_repository_files_exist() -> None:
    """Ensure the Stage 1 project contract is present."""
    root = Path(__file__).parents[1]
    required_files = (
        "README.md",
        "LICENSE",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        "configs/base.yaml",
        "configs/README.md",
        "docs/preprocessing.md",
        "scripts/smoke.py",
        "scripts/train_reference_models.py",
    )

    missing_files = [
        relative_path
        for relative_path in required_files
        if not (root / relative_path).is_file()
    ]

    assert not missing_files, f"Missing repository files: {missing_files}"
