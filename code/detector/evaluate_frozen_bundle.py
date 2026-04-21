from __future__ import annotations

"""
Evaluate a frozen detector bundle on a new standardized dataset.

This script is used for transfer experiments. It does not retrain anything.
It simply applies:

- frozen detector weights
- frozen calibration method
- frozen abstention threshold

to a new test CSV that has already been standardized in the required way.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _load_json(path: Path) -> Dict:
    """Load a frozen bundle or saved report."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _set_csv_field_limit() -> None:
    """Allow large CSV fields such as long retrieved evidence passages."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read the standardized transfer CSV into memory."""
    _set_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_to_xy(rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Project transfer rows onto the feature order required by the bundle."""
    x = np.array(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(row["judge_binary_label"]) for row in rows], dtype=np.int64)
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Convert detector logits to raw unsupported probabilities."""
    clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Negative log likelihood for calibrated or raw scores."""
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(-(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)).mean())


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AUROC only when both classes appear."""
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float | None]:
    """Compute the main test-set detector metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_at_threshold": float(y_pred.mean()),
    }


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE for the transfer evaluation."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def _build_curve_from_risk(y_true: np.ndarray, risk_scores: np.ndarray) -> List[Dict[str, float]]:
    """Build the transfer selective prediction curve from calibrated risk."""
    thresholds = sorted(set(float(score) for score in risk_scores.tolist()))
    thresholds = [0.0] + thresholds + [1.0]

    curve = []
    for threshold in thresholds:
        covered_mask = risk_scores <= threshold
        coverage = float(covered_mask.mean())
        if coverage == 0.0:
            curve.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "abstention_rate": 1.0,
                    "selective_risk": 0.0,
                    "selective_accuracy": 0.0,
                }
            )
            continue

        y_cov = y_true[covered_mask]
        selective_risk = float(y_cov.mean())
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "abstention_rate": float(1.0 - coverage),
                "selective_risk": selective_risk,
                "selective_accuracy": float(1.0 - selective_risk),
            }
        )
    return curve


def _plot_reliability_diagram(y_true: np.ndarray, probs: np.ndarray, output_path: Path, title: str) -> None:
    """Draw a reliability diagram for the transfer target set."""
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []

    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_conf.append(float(probs[mask].mean()))
        bin_acc.append(float(y_true[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(bin_conf, bin_acc, marker="o", linewidth=2, color="#0f4c81")
    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Observed Unsupported Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_curves(
    curve: Sequence[Dict[str, float]],
    operating_point: Dict[str, float],
    output_path_risk: Path,
    output_path_acc: Path,
    label: str,
) -> None:
    """Draw transfer risk-coverage and accuracy-coverage curves."""
    coverage = [point["coverage"] for point in curve]
    selective_risk = [point["selective_risk"] for point in curve]
    selective_accuracy = [point["selective_accuracy"] for point in curve]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(coverage, selective_risk, linewidth=2, color="#bb3e03", label=label)
    ax.scatter(
        [operating_point["coverage"]],
        [operating_point["selective_risk"]],
        color="black",
        label="Frozen Operating Point",
        zorder=3,
    )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path_risk.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_risk, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(coverage, selective_accuracy, linewidth=2, color="#0a9396", label=label)
    ax.scatter(
        [operating_point["coverage"]],
        [operating_point["selective_accuracy"]],
        color="black",
        label="Frozen Operating Point",
        zorder=3,
    )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Accuracy")
    ax.set_title("Accuracy-Coverage Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path_acc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_acc, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _apply_bundle_calibration(calibration: Dict, logits: np.ndarray, raw_probs: np.ndarray) -> np.ndarray:
    """Apply whichever calibration method was frozen inside the bundle."""
    method = calibration["method"].strip().lower()
    fit_parameters = calibration["fit_parameters"]
    if method == "platt scaling":
        return _sigmoid(
            float(fit_parameters["coef"]) * logits + float(fit_parameters["intercept"])
        )
    if method == "isotonic regression":
        # The bundle stores the method choice, but isotonic itself is piecewise and must be reconstructed
        # from the source report rather than from threshold count alone. For transfer, require the source
        # report to be used only when fit details are fully serialized.
        if "x_thresholds" in fit_parameters and "y_thresholds" in fit_parameters:
            model = IsotonicRegression(out_of_bounds="clip")
            model.X_thresholds_ = np.array(fit_parameters["x_thresholds"], dtype=np.float64)
            model.y_thresholds_ = np.array(fit_parameters["y_thresholds"], dtype=np.float64)
            model.f_ = lambda x: np.interp(
                x,
                model.X_thresholds_,
                model.y_thresholds_,
                left=model.y_thresholds_[0],
                right=model.y_thresholds_[-1],
            )
            return np.clip(model.predict(raw_probs), 0.0, 1.0)
        raise ValueError("Frozen bundle isotonic calibration is missing serialized thresholds.")
    raise ValueError(f"Unsupported calibration method in bundle: {calibration['method']}")


def main() -> None:
    """Evaluate a frozen source-domain detector on a target-domain dataset."""
    parser = argparse.ArgumentParser(description="Evaluate a frozen detector bundle on a new standardized dataset.")
    parser.add_argument("--bundle", required=True, help="Path to a frozen detector bundle JSON")
    parser.add_argument("--test", required=True, help="Standardized target-domain CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for target-domain evaluation outputs")
    parser.add_argument("--prefix", default="wikiqa_transfer")
    parser.add_argument("--source-name", default="phantom")
    parser.add_argument("--target-name", default="wikiqa")
    args = parser.parse_args()

    bundle = _load_json(Path(args.bundle))
    test_rows = _read_csv_rows(Path(args.test))

    feature_columns = bundle["feature_columns"]
    detector = bundle["detector"]
    calibration = bundle["calibration"]
    abstention_policy = bundle["abstention_policy"]

    x_test, y_test = _rows_to_xy(test_rows, feature_columns)
    weights = np.array([detector["coefficients"][column] for column in feature_columns], dtype=np.float64)
    logits = x_test @ weights + float(detector["intercept"])

    raw_probs = _sigmoid(logits)
    calibrated_probs = _apply_bundle_calibration(calibration, logits, raw_probs)

    classification_threshold = float(abstention_policy["classification_threshold_for_metrics"])
    frozen_threshold = float(abstention_policy["frozen_risk_threshold"])

    raw_metrics = {
        "nll": _nll(y_test, raw_probs),
        "ece": _expected_calibration_error(y_test, raw_probs),
        "brier": float(brier_score_loss(y_test, raw_probs)),
        "classification_metrics": _classification_metrics(y_test, raw_probs, classification_threshold),
    }
    calibrated_metrics = {
        "nll": _nll(y_test, calibrated_probs),
        "ece": _expected_calibration_error(y_test, calibrated_probs),
        "brier": float(brier_score_loss(y_test, calibrated_probs)),
        "classification_metrics": _classification_metrics(y_test, calibrated_probs, classification_threshold),
    }

    curve = _build_curve_from_risk(y_test, calibrated_probs)
    operating_point = min(curve, key=lambda point: abs(point["threshold"] - frozen_threshold))

    report = {
        "experiment_type": "frozen_bundle_transfer_evaluation",
        "source_name": args.source_name,
        "target_name": args.target_name,
        "bundle_path": str(Path(args.bundle)),
        "test_path": str(Path(args.test)),
        "feature_columns": feature_columns,
        "classification_threshold_for_metrics": classification_threshold,
        "frozen_risk_threshold": frozen_threshold,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "transfer_abstention": {
            "selection_rule": abstention_policy["selection_rule"],
            "frozen_source_threshold": frozen_threshold,
            "target_operating_point_at_frozen_threshold": operating_point,
            "target_curve": curve,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.prefix}_transfer_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    _plot_reliability_diagram(
        y_test,
        calibrated_probs,
        output_dir / f"{args.prefix}_reliability_diagram.png",
        title=f"Reliability Diagram on {args.target_name.title()}",
    )
    _plot_coverage_curves(
        curve,
        operating_point,
        output_dir / f"{args.prefix}_risk_coverage_curve.png",
        output_dir / f"{args.prefix}_accuracy_coverage_curve.png",
        label=args.target_name.title(),
    )

    print(f"Saved transfer evaluation report to: {report_path}")
    print(
        json.dumps(
            {
                "source_name": args.source_name,
                "target_name": args.target_name,
                "frozen_risk_threshold": frozen_threshold,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "target_operating_point_at_frozen_threshold": operating_point,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
