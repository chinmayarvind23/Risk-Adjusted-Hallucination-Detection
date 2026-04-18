from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


FEATURE_COLUMNS = [
    "mean_token_nll",
    "self_consistency_disagreement",
    "semantic_entropy",
    "groundedness_score",
]

FEATURE_LABELS = {
    "mean_token_nll": "Mean Token NLL",
    "self_consistency_disagreement": "Self-Consistency Disagreement",
    "semantic_entropy": "Semantic Entropy",
    "groundedness_score": "Groundedness Score",
}

METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

BASELINE_PLOT_ORDER = [
    "all_four_features",
    "self_consistency_only",
    "token_only",
    "semantic_entropy_only",
    "groundedness_only",
    "manual_weighted",
    "random_score",
]

BASELINE_DISPLAY_NAMES = {
    "all_four_features": "Full 4-Feature Detector",
    "self_consistency_only": "Self-Consistency Only",
    "token_only": "Token Uncertainty Only",
    "semantic_entropy_only": "Semantic Entropy Only",
    "groundedness_only": "Groundedness Only",
    "manual_weighted": "Manual Weighted",
    "random_score": "Random",
}

BASELINE_COLORS = {
    "all_four_features": "#0f4c81",
    "self_consistency_only": "#0a9396",
    "token_only": "#ee9b00",
    "semantic_entropy_only": "#ca6702",
    "groundedness_only": "#bb3e03",
    "manual_weighted": "#6a4c93",
    "random_score": "#9aa5b1",
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_to_xy(rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(row["judge_binary_label"]) for row in rows], dtype=np.int64)
    return x, y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _manual_weighted_scores(x: np.ndarray) -> np.ndarray:
    raw_score = 0.10 * x[:, 0] + 0.20 * x[:, 1] + 0.30 * x[:, 2] - 0.40 * x[:, 3]
    return _sigmoid(raw_score)


def _reconstruct_logistic_scores(x: np.ndarray, coefficients: Dict[str, float], intercept: float) -> np.ndarray:
    ordered_weights = np.array([coefficients[column] for column in coefficients.keys()], dtype=np.float64)
    return _sigmoid(x @ ordered_weights + intercept)


def _extract_best_trials(baseline_report: Dict) -> Dict[str, Dict]:
    best_trials: Dict[str, Dict] = {}
    for name, payload in baseline_report["baselines"].items():
        best_trials[name] = payload["best_trial"]
    return best_trials


def _plot_baseline_metric_bars(baseline_report: Dict, output_path: Path) -> None:
    best_trials = _extract_best_trials(baseline_report)
    metrics = ["auroc", "auprc", "f1", "accuracy"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), constrained_layout=True)

    ordered_names = [name for name in BASELINE_PLOT_ORDER if name in best_trials]
    display_names = [BASELINE_DISPLAY_NAMES[name] for name in ordered_names]
    x = np.arange(len(ordered_names))

    for ax, metric_name in zip(axes, metrics):
        values = [best_trials[name]["test_metrics"][metric_name] for name in ordered_names]
        colors = [BASELINE_COLORS[name] for name in ordered_names]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(METRIC_LABELS[metric_name])
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("PHANTOM Test Performance Across Detector Baselines")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_metric_heatmap(baseline_report: Dict, output_path: Path) -> None:
    best_trials = _extract_best_trials(baseline_report)
    ordered_names = [name for name in BASELINE_PLOT_ORDER if name in best_trials]
    metrics = ["auroc", "auprc", "accuracy", "precision", "recall", "f1"]

    matrix = np.array(
        [[best_trials[name]["test_metrics"][metric] for metric in metrics] for name in ordered_names],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ordered_names)))
    ax.set_yticklabels([BASELINE_DISPLAY_NAMES[name] for name in ordered_names])
    ax.set_title("Baseline Comparison Heatmap on PHANTOM Test Set")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.9, label="Score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tuned_coefficients(tuned_report: Dict, output_path: Path) -> None:
    best_trial = tuned_report["best_trial"]
    coeffs = best_trial["coefficients"]
    features = list(coeffs.keys())
    values = [coeffs[name] for name in features]
    colors = ["#b22222" if value < 0 else "#0f4c81" for value in values]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(np.arange(len(features)), values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([FEATURE_LABELS.get(name, name) for name in features], rotation=20, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("Learned Logistic Regression Weights")
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(values):
        y = value + (0.02 if value >= 0 else -0.05)
        va = "bottom" if value >= 0 else "top"
        ax.text(idx, y, f"{value:.3f}", ha="center", va=va, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_distributions(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    supported_rows = [row for row in rows if int(row["judge_binary_label"]) == 0]
    unsupported_rows = [row for row in rows if int(row["judge_binary_label"]) == 1]

    for ax, feature_name in zip(axes, FEATURE_COLUMNS):
        supported = np.array([float(row[feature_name]) for row in supported_rows], dtype=np.float64)
        unsupported = np.array([float(row[feature_name]) for row in unsupported_rows], dtype=np.float64)

        ax.hist(supported, bins=30, alpha=0.6, density=True, label="Supported", color="#0a9396")
        ax.hist(unsupported, bins=30, alpha=0.6, density=True, label="Unsupported", color="#bb3e03")
        ax.set_title(FEATURE_LABELS[feature_name])
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle("Standardized Feature Distributions by Judge Label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_correlation(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    x, _ = _rows_to_xy(rows, FEATURE_COLUMNS)
    corr = np.corrcoef(x, rowvar=False)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_COLUMNS], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_COLUMNS])
    ax.set_title("Feature Correlation Heatmap")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.9, label="Correlation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_pr_curves(
    baseline_report: Dict,
    tuned_report: Dict,
    test_rows: Sequence[Dict[str, str]],
    output_path: Path,
) -> None:
    best_trials = _extract_best_trials(baseline_report)
    x_all, y_test = _rows_to_xy(test_rows, FEATURE_COLUMNS)

    curve_specs: List[Tuple[str, np.ndarray]] = []

    if "all_four_features" in best_trials:
        trial = best_trials["all_four_features"]
        scores = _reconstruct_logistic_scores(
            x_all,
            coefficients=trial["coefficients"],
            intercept=float(trial["intercept"]),
        )
        curve_specs.append(("Full 4-Feature Detector", scores))

    if "self_consistency_only" in best_trials:
        trial = best_trials["self_consistency_only"]
        x_single, _ = _rows_to_xy(test_rows, trial["feature_columns"])
        scores = _reconstruct_logistic_scores(
            x_single,
            coefficients=trial["coefficients"],
            intercept=float(trial["intercept"]),
        )
        curve_specs.append(("Self-Consistency Only", scores))

    if "token_only" in best_trials:
        trial = best_trials["token_only"]
        x_single, _ = _rows_to_xy(test_rows, trial["feature_columns"])
        scores = _reconstruct_logistic_scores(
            x_single,
            coefficients=trial["coefficients"],
            intercept=float(trial["intercept"]),
        )
        curve_specs.append(("Token Only", scores))

    if "manual_weighted" in best_trials:
        curve_specs.append(("Manual Weighted", _manual_weighted_scores(x_all)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    roc_ax = axes[0]
    for name, scores in curve_specs:
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_ax.plot(fpr, tpr, linewidth=2, label=name)
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    roc_ax.set_title("ROC Curves on PHANTOM Test Set")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.grid(alpha=0.25)
    roc_ax.legend()

    pr_ax = axes[1]
    for name, scores in curve_specs:
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_ax.plot(recall, precision, linewidth=2, label=name)
    pr_ax.set_title("Precision-Recall Curves on PHANTOM Test Set")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.grid(alpha=0.25)
    pr_ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_plot_guide(output_dir: Path, prefix: str) -> None:
    lines = [
        "# Plot Guide",
        "",
        "## Best plots for the poster",
        "",
        "1. Baseline metric bars",
        "2. Logistic regression coefficient plot",
        "3. ROC and PR curves",
        "",
        "## Good extra plots for the report",
        "",
        "1. Baseline heatmap",
        "2. Feature distributions by label",
        "3. Feature correlation heatmap",
        "",
        "## Files generated",
        "",
        f"- {prefix}_baseline_metric_bars.png",
        f"- {prefix}_baseline_metric_heatmap.png",
        f"- {prefix}_tuned_logreg_coefficients.png",
        f"- {prefix}_roc_pr_curves.png",
        f"- {prefix}_feature_distributions.png",
        f"- {prefix}_feature_correlation.png",
    ]

    guide_path = output_dir / f"{prefix}_plot_guide.md"
    guide_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detector and baseline plots for the report and poster.")
    parser.add_argument("--baseline-report", required=True, help="Path to baseline report JSON")
    parser.add_argument("--tuned-report", required=True, help="Path to tuned logreg report JSON")
    parser.add_argument("--train", required=True, help="Standardized train CSV")
    parser.add_argument("--val", required=True, help="Standardized validation CSV")
    parser.add_argument("--test", required=True, help="Standardized test CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for plot PNGs")
    parser.add_argument("--prefix", default="phantom_4000")
    args = parser.parse_args()

    baseline_report = _load_json(Path(args.baseline_report))
    tuned_report = _load_json(Path(args.tuned_report))

    train_rows = _read_csv_rows(Path(args.train))
    val_rows = _read_csv_rows(Path(args.val))
    test_rows = _read_csv_rows(Path(args.test))
    combined_rows = train_rows + val_rows + test_rows

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_baseline_metric_bars(
        baseline_report,
        output_dir / f"{args.prefix}_baseline_metric_bars.png",
    )
    _plot_baseline_metric_heatmap(
        baseline_report,
        output_dir / f"{args.prefix}_baseline_metric_heatmap.png",
    )
    _plot_tuned_coefficients(
        tuned_report,
        output_dir / f"{args.prefix}_tuned_logreg_coefficients.png",
    )
    _plot_roc_pr_curves(
        baseline_report,
        tuned_report,
        test_rows,
        output_dir / f"{args.prefix}_roc_pr_curves.png",
    )
    _plot_feature_distributions(
        combined_rows,
        output_dir / f"{args.prefix}_feature_distributions.png",
    )
    _plot_feature_correlation(
        combined_rows,
        output_dir / f"{args.prefix}_feature_correlation.png",
    )
    _write_plot_guide(output_dir, args.prefix)

    print(f"Saved plot set to: {output_dir}")
    print(f"- {args.prefix}_baseline_metric_bars.png")
    print(f"- {args.prefix}_baseline_metric_heatmap.png")
    print(f"- {args.prefix}_tuned_logreg_coefficients.png")
    print(f"- {args.prefix}_roc_pr_curves.png")
    print(f"- {args.prefix}_feature_distributions.png")
    print(f"- {args.prefix}_feature_correlation.png")
    print(f"- {args.prefix}_plot_guide.md")


if __name__ == "__main__":
    main()
