from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METRIC_ORDER = ["auroc", "auprc", "accuracy", "precision", "recall", "f1"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_method_metrics(report: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    methods = {
        "Baseline LogReg": report["logistic_regression"],
        "Manual Weighted": report["manual_weighted_ablation"],
    }
    if "hyperparameter_tuning" in report and report["hyperparameter_tuning"].get("best_trial"):
        methods["Tuned LogReg"] = report["hyperparameter_tuning"]["best_trial"]
    return methods


def _plot_metric_comparison(report: Dict, output_path: Path) -> None:
    methods = _extract_method_metrics(report)
    method_names = list(methods.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, split_name in zip(axes, ["val_metrics", "test_metrics"]):
        x = np.arange(len(METRIC_ORDER))
        width = 0.8 / max(len(method_names), 1)

        for idx, method_name in enumerate(method_names):
            metric_block = methods[method_name][split_name]
            values = [metric_block.get(metric, 0.0) or 0.0 for metric in METRIC_ORDER]
            ax.bar(x + (idx - (len(method_names) - 1) / 2) * width, values, width=width, label=method_name)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER], rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{split_name.replace('_', ' ').title()}")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Detector Performance Comparison")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_coefficients(report: Dict, output_path: Path) -> None:
    coeffs = report["logistic_regression"]["coefficients"]
    tuned_coeffs = None
    if "hyperparameter_tuning" in report and report["hyperparameter_tuning"].get("best_trial"):
        tuned_coeffs = report["hyperparameter_tuning"]["best_trial"]["coefficients"]

    features = list(coeffs.keys())
    baseline_vals = [coeffs[f] for f in features]

    x = np.arange(len(features))
    width = 0.35 if tuned_coeffs else 0.6

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - (width / 2 if tuned_coeffs else 0), baseline_vals, width=width, label="Baseline LogReg")

    if tuned_coeffs:
        tuned_vals = [tuned_coeffs[f] for f in features]
        ax.bar(x + width / 2, tuned_vals, width=width, label="Tuned LogReg")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("Learned Logistic Regression Feature Weights")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot detector metrics and coefficients from a logreg report JSON.")
    parser.add_argument("--report", required=True, help="Path to phantom_4000_logreg_report.json")
    parser.add_argument("--output-dir", required=True, help="Directory for output PNGs")
    parser.add_argument("--prefix", default="phantom_4000")
    args = parser.parse_args()

    report = _load_json(Path(args.report))
    output_dir = Path(args.output_dir)

    metrics_plot = output_dir / f"{args.prefix}_detector_metrics.png"
    coeff_plot = output_dir / f"{args.prefix}_detector_coefficients.png"

    _plot_metric_comparison(report, metrics_plot)
    _plot_coefficients(report, coeff_plot)

    print(f"Saved metric comparison plot to: {metrics_plot}")
    print(f"Saved coefficient plot to: {coeff_plot}")


if __name__ == "__main__":
    main()
