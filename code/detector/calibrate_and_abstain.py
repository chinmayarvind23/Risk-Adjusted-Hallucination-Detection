from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_COLUMNS = [
    "mean_token_nll",
    "self_consistency_disagreement",
    "semantic_entropy",
    "groundedness_score",
]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_to_xy(rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=np.float64,
    )
    y = np.array([int(row["judge_binary_label"]) for row in rows], dtype=np.int64)
    return x, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(-(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)).mean())


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float | None]:
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
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def _fit_temperature(val_logits: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    coarse_grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 600))
    coarse_losses = []
    for temp in coarse_grid:
        probs = _sigmoid(val_logits / temp)
        coarse_losses.append(_nll(y_val, probs))
    best_idx = int(np.argmin(coarse_losses))
    best_temp = float(coarse_grid[best_idx])

    lower = max(0.01, best_temp / 2.0)
    upper = best_temp * 2.0
    fine_grid = np.exp(np.linspace(np.log(lower), np.log(upper), 600))
    fine_losses = []
    for temp in fine_grid:
        probs = _sigmoid(val_logits / temp)
        fine_losses.append(_nll(y_val, probs))
    fine_best_idx = int(np.argmin(fine_losses))
    fine_best_temp = float(fine_grid[fine_best_idx])
    fine_best_loss = float(fine_losses[fine_best_idx])

    return {
        "temperature": fine_best_temp,
        "validation_nll_after_temperature_scaling": fine_best_loss,
    }


def _build_curve_from_risk(y_true: np.ndarray, risk_scores: np.ndarray) -> List[Dict[str, float]]:
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
        selective_accuracy = float(1.0 - selective_risk)
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "abstention_rate": float(1.0 - coverage),
                "selective_risk": selective_risk,
                "selective_accuracy": selective_accuracy,
            }
        )
    return curve


def _select_operating_point(curve: Sequence[Dict[str, float]], min_coverage: float) -> Dict[str, float]:
    eligible = [point for point in curve if point["coverage"] >= min_coverage]
    if not eligible:
        eligible = list(curve)

    eligible.sort(
        key=lambda point: (
            point["selective_accuracy"],
            -point["selective_risk"],
            point["coverage"],
        ),
        reverse=True,
    )
    return eligible[0]


def _plot_reliability_diagram(y_true: np.ndarray, probs: np.ndarray, output_path: Path, title: str) -> None:
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    bin_conf = []

    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_centers.append((left + right) / 2.0)
        bin_acc.append(float(y_true[mask].mean()))
        bin_conf.append(float(probs[mask].mean()))

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
    val_curve: Sequence[Dict[str, float]],
    test_curve: Sequence[Dict[str, float]],
    operating_point: Dict[str, float],
    output_path_risk: Path,
    output_path_acc: Path,
) -> None:
    val_cov = [point["coverage"] for point in val_curve]
    val_risk = [point["selective_risk"] for point in val_curve]
    val_acc = [point["selective_accuracy"] for point in val_curve]

    test_cov = [point["coverage"] for point in test_curve]
    test_risk = [point["selective_risk"] for point in test_curve]
    test_acc = [point["selective_accuracy"] for point in test_curve]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(val_cov, val_risk, label="Validation", linewidth=2, color="#0a9396")
    ax.plot(test_cov, test_risk, label="Test", linewidth=2, color="#bb3e03")
    ax.scatter(
        [operating_point["coverage"]],
        [operating_point["selective_risk"]],
        color="black",
        label="Selected Operating Point",
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
    ax.plot(val_cov, val_acc, label="Validation", linewidth=2, color="#0a9396")
    ax.plot(test_cov, test_acc, label="Test", linewidth=2, color="#bb3e03")
    ax.scatter(
        [operating_point["coverage"]],
        [operating_point["selective_accuracy"]],
        color="black",
        label="Selected Operating Point",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a tuned detector with temperature scaling and evaluate abstention.")
    parser.add_argument("--tuned-report", required=True, help="Path to tuned logreg report JSON")
    parser.add_argument("--val", required=True, help="Standardized validation CSV")
    parser.add_argument("--test", required=True, help="Standardized test CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for calibration outputs")
    parser.add_argument("--prefix", default="phantom_4000")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum validation coverage required when selecting the abstention threshold.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.50,
        help="Threshold for converting calibrated risk into a binary unsupported prediction when reporting classification metrics.",
    )
    args = parser.parse_args()

    tuned_report = _load_json(Path(args.tuned_report))
    best_trial = tuned_report["best_trial"]
    coefficients = best_trial["coefficients"]
    intercept = float(best_trial["intercept"])
    feature_columns = list(coefficients.keys())

    val_rows = _read_csv_rows(Path(args.val))
    test_rows = _read_csv_rows(Path(args.test))
    x_val, y_val = _rows_to_xy(val_rows, feature_columns)
    x_test, y_test = _rows_to_xy(test_rows, feature_columns)

    weights = np.array([coefficients[column] for column in feature_columns], dtype=np.float64)
    val_logits = x_val @ weights + intercept
    test_logits = x_test @ weights + intercept

    raw_val_probs = _sigmoid(val_logits)
    raw_test_probs = _sigmoid(test_logits)

    temperature_fit = _fit_temperature(val_logits, y_val)
    temperature = float(temperature_fit["temperature"])

    calibrated_val_probs = _sigmoid(val_logits / temperature)
    calibrated_test_probs = _sigmoid(test_logits / temperature)

    raw_metrics = {
        "val": {
            "nll": _nll(y_val, raw_val_probs),
            "ece": _expected_calibration_error(y_val, raw_val_probs),
            "brier": float(brier_score_loss(y_val, raw_val_probs)),
            "classification_metrics": _classification_metrics(y_val, raw_val_probs, args.classification_threshold),
        },
        "test": {
            "nll": _nll(y_test, raw_test_probs),
            "ece": _expected_calibration_error(y_test, raw_test_probs),
            "brier": float(brier_score_loss(y_test, raw_test_probs)),
            "classification_metrics": _classification_metrics(y_test, raw_test_probs, args.classification_threshold),
        },
    }

    calibrated_metrics = {
        "val": {
            "nll": _nll(y_val, calibrated_val_probs),
            "ece": _expected_calibration_error(y_val, calibrated_val_probs),
            "brier": float(brier_score_loss(y_val, calibrated_val_probs)),
            "classification_metrics": _classification_metrics(y_val, calibrated_val_probs, args.classification_threshold),
        },
        "test": {
            "nll": _nll(y_test, calibrated_test_probs),
            "ece": _expected_calibration_error(y_test, calibrated_test_probs),
            "brier": float(brier_score_loss(y_test, calibrated_test_probs)),
            "classification_metrics": _classification_metrics(y_test, calibrated_test_probs, args.classification_threshold),
        },
    }

    val_curve = _build_curve_from_risk(y_val, calibrated_val_probs)
    test_curve = _build_curve_from_risk(y_test, calibrated_test_probs)
    operating_point = _select_operating_point(val_curve, min_coverage=args.min_coverage)
    frozen_threshold = float(operating_point["threshold"])

    selected_test_point = min(
        test_curve,
        key=lambda point: abs(point["threshold"] - frozen_threshold),
    )

    report = {
        "source": "phantom",
        "feature_columns": feature_columns,
        "min_coverage_constraint": args.min_coverage,
        "classification_threshold_for_metrics": args.classification_threshold,
        "tuned_detector": {
            "coefficients": coefficients,
            "intercept": intercept,
        },
        "temperature_scaling": {
            **temperature_fit,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
        },
        "abstention": {
            "selection_rule": "Choose the validation risk threshold that maximizes selective accuracy subject to the minimum coverage constraint.",
            "frozen_threshold": frozen_threshold,
            "selected_validation_operating_point": operating_point,
            "test_operating_point_at_frozen_threshold": selected_test_point,
            "validation_curve": val_curve,
            "test_curve": test_curve,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.prefix}_calibration_abstention_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    _plot_reliability_diagram(
        y_val,
        calibrated_val_probs,
        output_dir / f"{args.prefix}_reliability_diagram_val.png",
        title="Reliability Diagram on Validation Set",
    )
    _plot_reliability_diagram(
        y_test,
        calibrated_test_probs,
        output_dir / f"{args.prefix}_reliability_diagram_test.png",
        title="Reliability Diagram on Test Set",
    )
    _plot_coverage_curves(
        val_curve,
        test_curve,
        operating_point,
        output_dir / f"{args.prefix}_risk_coverage_curve.png",
        output_dir / f"{args.prefix}_accuracy_coverage_curve.png",
    )

    print(f"Saved calibration and abstention report to: {report_path}")
    print(
        json.dumps(
            {
                "temperature": temperature,
                "validation_ece_before": raw_metrics["val"]["ece"],
                "validation_ece_after": calibrated_metrics["val"]["ece"],
                "validation_brier_before": raw_metrics["val"]["brier"],
                "validation_brier_after": calibrated_metrics["val"]["brier"],
                "test_ece_before": raw_metrics["test"]["ece"],
                "test_ece_after": calibrated_metrics["test"]["ece"],
                "test_brier_before": raw_metrics["test"]["brier"],
                "test_brier_after": calibrated_metrics["test"]["brier"],
                "frozen_threshold": frozen_threshold,
                "selected_validation_operating_point": operating_point,
                "test_operating_point_at_frozen_threshold": selected_test_point,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
