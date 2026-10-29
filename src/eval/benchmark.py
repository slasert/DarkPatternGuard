"""Read-only benchmark, calibration, learning-curve, and error reporting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import configured_seeds
from .metrics import LABELS, PROBABILITY_COLUMNS


def _prediction_files(results_dir: Path) -> list[tuple[str, int, Path]]:
    files = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for seed_dir in sorted(model_dir.glob("seed*")):
            match = re.fullmatch(r"seed(\d+)", seed_dir.name)
            if match and (seed_dir / "preds.csv").is_file():
                files.append(
                    (model_dir.name, int(match.group(1)), seed_dir / "preds.csv")
                )
    return files


def _metrics(path: Path) -> dict:
    metrics_path = path.parent / "metrics.json"
    return (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )


def write_leaderboard(results_dir: Path = Path("results")) -> pd.DataFrame:
    """Aggregate saved metrics across seeds and write Markdown/CSV tables."""
    records = []
    for model, seed, path in _prediction_files(results_dir):
        metrics = _metrics(path)
        records.append(
            {
                "model": model,
                "seed": seed,
                "macro_f1": metrics.get("macro_f1", np.nan),
                "macro_f1_ci_lower": metrics.get("macro_f1_bootstrap_95_ci", {}).get(
                    "lower", np.nan
                ),
                "macro_f1_ci_upper": metrics.get("macro_f1_bootstrap_95_ci", {}).get(
                    "upper", np.nan
                ),
                "params_count": metrics.get("params_count", np.nan),
                "train_time_seconds": metrics.get("train_time_seconds", np.nan),
                "inference_ms_per_sample": metrics.get(
                    "inference_ms_per_sample", np.nan
                ),
                "llm_usd_per_1k_texts": metrics.get("llm_cost_latency", {}).get(
                    "estimated_cost_usd", np.nan
                )
                * 1000
                / max(1, len(pd.read_csv(path))),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FileNotFoundError(f"No saved predictions found in {results_dir}")
    summary = (
        frame.groupby("model", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_ci_lower=("macro_f1_ci_lower", "mean"),
            mean_ci_upper=("macro_f1_ci_upper", "mean"),
            params_count=("params_count", "mean"),
            train_time_seconds=("train_time_seconds", "mean"),
            inference_ms_per_sample=("inference_ms_per_sample", "mean"),
            llm_usd_per_1k_texts=("llm_usd_per_1k_texts", "mean"),
        )
        .fillna("")
    )
    summary["mean_pm_std"] = summary.apply(
        lambda row: (
            f"{row['mean_macro_f1']:.4f} ± {row['std_macro_f1']:.4f}"
            if row["std_macro_f1"] != ""
            else f"{row['mean_macro_f1']:.4f}"
        ),
        axis=1,
    )
    summary.to_csv(results_dir / "leaderboard.csv", index=False)
    (results_dir / "leaderboard.md").write_text(
        summary.to_markdown(index=False), encoding="utf-8"
    )
    return summary


def write_calibration_plot(results_dir: Path) -> None:
    """Plot confidence versus accuracy from saved prediction probabilities."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for model, _, path in _prediction_files(results_dir):
        frame = pd.read_csv(path)
        probability_columns = list(PROBABILITY_COLUMNS)
        probabilities = frame[probability_columns].to_numpy()
        confidence = probabilities.max(axis=1)
        correct = (
            np.asarray(LABELS)[probabilities.argmax(axis=1)] == frame["true_label"]
        )
        bins = np.linspace(0, 1, 11)
        centers, accuracies = [], []
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            selected = (confidence > lower) & (confidence <= upper)
            if selected.any():
                centers.append(confidence[selected].mean())
                accuracies.append(correct[selected].mean())
        axis.plot(centers, accuracies, marker="o", label=model)
    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration"
    )
    axis.set(xlabel="Mean confidence", ylabel="Accuracy", title="Calibration curves")
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(results_dir / "calibration.png", dpi=150)
    plt.close(figure)


def write_learning_curve(results_dir: Path) -> None:
    """Plot Macro-F1 versus available training-size metadata."""
    rows = []
    for model, _, path in _prediction_files(results_dir):
        metrics = _metrics(path)
        if "train_size" in metrics:
            rows.append((model, metrics["train_size"], metrics.get("macro_f1", np.nan)))
    figure, axis = plt.subplots(figsize=(7, 6))
    if rows:
        frame = pd.DataFrame(rows, columns=("model", "train_size", "macro_f1"))
        for model, group in frame.groupby("model"):
            axis.plot(group["train_size"], group["macro_f1"], marker="o", label=model)
        axis.set(xlabel="Training rows", ylabel="Macro-F1", title="Learning curve")
        axis.legend(fontsize="small")
    else:
        axis.text(
            0.5,
            0.5,
            "Training-size metadata unavailable in saved metrics",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        axis.set_title("Learning curve")
    figure.tight_layout()
    figure.savefig(results_dir / "learning_curve.png", dpi=150)
    plt.close(figure)


def error_rows(frame: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Shared row-level errors, preserving metadata for categorized analysis."""
    errors = frame.loc[predicted != frame["true_label"]].copy()
    errors["predicted"] = predicted[predicted != frame["true_label"]]
    text = errors["text"] if "text" in errors else errors["id"]
    errors["text_length"] = text.fillna(errors["id"]).astype(str).str.len()
    return errors


def write_error_analysis(results_dir: Path = Path("results")) -> None:
    """Group all saved model errors and highlight baseline-vs-novel wins."""
    prediction_sets = {
        model: pd.read_csv(path)
        for model, seed, path in _prediction_files(results_dir)
        if seed == configured_seeds()[0]
    }
    if not prediction_sets:
        raise FileNotFoundError("No predictions available for error analysis")
    for frame in prediction_sets.values():
        frame["id"] = frame["id"].astype(str)
    test_path = Path("data/cleaned/test.csv")
    source = None
    if test_path.is_file():
        source = pd.read_csv(test_path)
        source["id"] = source.index.astype(str)
        source = source[["id", "text"]]
        prediction_sets = {
            model: frame.merge(source, on="id", how="left")
            for model, frame in prediction_sets.items()
        }
    lines = [
        "# Error analysis",
        "",
        "## Errors by model, class pair, and text length",
        "",
    ]
    for model, frame in prediction_sets.items():
        predicted = np.asarray(LABELS)[
            frame[list(PROBABILITY_COLUMNS)].to_numpy().argmax(axis=1)
        ]
        errors = error_rows(frame, predicted)
        lines.append(f"### {model}")
        if errors.empty:
            lines.append("No errors found.")
            continue
        grouped = (
            errors.groupby(["true_label", "predicted"]).size().reset_index(name="count")
        )
        lines.append(grouped.to_markdown(index=False))
        lines.append("")
        bins = pd.cut(
            errors["text_length"],
            bins=[0, 40, 100, np.inf],
            labels=["short", "medium", "long"],
        )
        lines.append(
            errors.assign(length_bin=bins)
            .groupby("length_bin", observed=False)
            .size()
            .to_frame("count")
            .to_markdown()
        )
        lines.append("")
    baseline = prediction_sets.get("finetune_bert")
    novel = prediction_sets.get("finetune_deberta")
    llm = prediction_sets.get("llm_gpt4o_mini")
    if baseline is not None:
        lines.extend(["## Baseline failures recovered by novel models", ""])
        candidates: dict[tuple[str, str, str, str, str, str], list[int]] = {}
        alternatives = (
            ("DeBERTa", "finetune_deberta", novel),
            ("GPT-4o-mini", "llm_gpt4o_mini", llm),
        )
        for alternative_name, model_name, alternative in alternatives:
            if alternative is None:
                continue
            for seed in configured_seeds():
                baseline_path = (
                    results_dir / "finetune_bert" / f"seed{seed}" / "preds.csv"
                )
                alternative_path = (
                    results_dir / model_name / f"seed{seed}" / "preds.csv"
                )
                if not baseline_path.is_file() or not alternative_path.is_file():
                    continue
                baseline_seed = pd.read_csv(baseline_path)
                alternative_seed = pd.read_csv(alternative_path)
                baseline_seed["id"] = baseline_seed["id"].astype(str)
                alternative_seed["id"] = alternative_seed["id"].astype(str)
                aligned = baseline_seed.merge(
                    alternative_seed,
                    on="id",
                    how="inner",
                    suffixes=("_bert", "_alternative"),
                    validate="one_to_one",
                )
                if aligned.empty:
                    raise ValueError(
                        f"{alternative_name} and BERT have no shared test IDs "
                        f"for seed {seed}"
                    )
                if not np.array_equal(
                    aligned["true_label_bert"], aligned["true_label_alternative"]
                ):
                    raise ValueError(
                        f"{alternative_name} and BERT disagree on shared test labels "
                        f"for seed {seed}"
                    )
                if source is not None:
                    aligned = aligned.merge(
                        source, on="id", how="left", validate="many_to_one"
                    )
                true_labels = aligned["true_label_bert"].to_numpy()
                baseline_pred = np.asarray(LABELS)[
                    aligned[[f"{column}_bert" for column in PROBABILITY_COLUMNS]]
                    .to_numpy()
                    .argmax(axis=1)
                ]
                alternative_pred = np.asarray(LABELS)[
                    aligned[[f"{column}_alternative" for column in PROBABILITY_COLUMNS]]
                    .to_numpy()
                    .argmax(axis=1)
                ]
                recovered = (baseline_pred != true_labels) & (
                    alternative_pred == true_labels
                )
                for index in np.flatnonzero(recovered):
                    row = aligned.iloc[index]
                    text = (
                        str(row["text"])
                        if "text" in aligned.columns and pd.notna(row["text"])
                        else "(text unavailable in saved results)"
                    )
                    text = (
                        text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
                    )
                    candidate = (
                        alternative_name,
                        row["id"],
                        text,
                        true_labels[index],
                        baseline_pred[index],
                        alternative_pred[index],
                    )
                    candidates.setdefault(candidate, []).append(seed)
        lines.append(
            "| Alternative | ID | Original text | True label | "
            "BERT prediction | Alternative prediction | Seed(s) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for candidate, seeds in list(candidates.items())[:10]:
            row = (*candidate, ", ".join(map(str, seeds)))
            lines.append("| " + " | ".join(map(str, row)) + " |")
        if len(candidates) < 5:
            lines.append("")
            lines.append(
                f"Only {len(candidates)} recovered baseline errors were present "
                "in saved results."
            )
    (results_dir / "errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all available benchmark artifacts from saved outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    write_leaderboard(args.results_dir)
    write_calibration_plot(args.results_dir)
    write_learning_curve(args.results_dir)
    write_error_analysis(args.results_dir)


if __name__ == "__main__":
    main()
