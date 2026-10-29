"""Audit README coverage and forbidden tracked or present artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
FORBIDDEN_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache_d",
    "cache",
    "checkpoints",
    "models",
    "models_out",
    "models_cache",
    "huggingface_cache",
    "experiment_runs",
    "test-results",
    "playwright-report",
    "dist",
    "build",
}


def files_on_disk() -> list[Path]:
    """Return source/document files while skipping generated dependency trees."""
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
    ]


def tracked_files() -> tuple[bool, list[Path]]:
    """Return whether Git is available and its tracked files."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, []
    return True, [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    """Print audit findings and return non-zero only for forbidden artifacts."""
    readme_text = README.read_text(encoding="utf-8")
    present_forbidden = [
        path for path in files_on_disk() if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    env_files = [path for path in files_on_disk() if path.name == ".env"]
    git_available, tracked = tracked_files()
    tracked_cache_files = [
        path for path in tracked if ".cache_d" in path.relative_to(ROOT).parts
    ]
    tracked_forbidden = [
        path
        for path in tracked
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
        or path.name == ".env"
        or path in tracked_cache_files
    ]
    documented_candidates = [
        path
        for path in files_on_disk()
        if path.suffix in {".py", ".md", ".yaml", ".yml", ".ts", ".tsx", ".json"}
        and path.name != README.name
        and path.name not in {"package-lock.json", "tsconfig.tsbuildinfo"}
    ]
    normalized_readme = readme_text.replace("\\", "/")
    undocumented = [
        path.relative_to(ROOT)
        for path in documented_candidates
        if str(path.relative_to(ROOT)).replace("\\", "/") not in normalized_readme
    ]

    print(f"README: {README}")
    print(f"Git repository detected: {git_available}")
    if not git_available:
        print("Git tracking check: SKIPPED (workspace is not a Git checkout)")
    else:
        print(
            f"Tracked forbidden files or .cache_d contents: "
            f"{len(tracked_forbidden)}"
        )
        for path in tracked_forbidden:
            print(f"  FORBIDDEN TRACKED: {path.relative_to(ROOT)}")
    print(
        "Forbidden weights outside ignored local artifact/cache directories: "
        f"{len(present_forbidden)}"
    )
    for path in present_forbidden:
        print(f"  FORBIDDEN PRESENT: {path.relative_to(ROOT)}")
    print(f".env files present on disk: {len(env_files)}")
    print(f"README-unreferenced candidate files: {len(undocumented)}")
    for path in undocumented:
        print(f"  UNREFERENCED: {path}")
    return 1 if tracked_forbidden or present_forbidden or env_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
