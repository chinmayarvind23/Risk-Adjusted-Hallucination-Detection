from __future__ import annotations

"""
Apply the final calibration method, build abstention curves, and save a frozen bundle.

This is the final in-domain reliability script. It takes the tuned detector,
fits a calibrator on validation only, and then:

1. reports raw and calibrated metrics
2. builds risk-coverage and accuracy-coverage curves
3. chooses a validation operating point under a minimum coverage constraint
4. freezes that threshold
5. evaluates the frozen threshold on test
6. writes a reusable bundle for later transfer experiments
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
from sklearn.linear_model import LogisticRegression
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


def _infer_source_name(prefix: str, tuned_report_path: Path, val_path: Path, test_path: Path) -> str:
    """Infer a dataset name for saved reports and bundles."""
    joined = " ".join(
        [
            prefix.lower(),
            tuned_report_path.name.lower(),
            str(val_path).lower(),
            str(test_path).lower(),
        ]
    )
    if "wikiqa" in joined or "wiki_qa" in joined:
        return "wikiqa"
    if "phantom" in joined:
        return "phantom"
    return "unknown"


def _set_csv_field_limit() -> None:
    """Allow large CSV fields such as retained context text."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a split CSV into memory."""
    _set_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> Dict:
    """Load a JSON file produced by an earlier pipeline stage."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows_to_xy(rows: Sequence[Dict[str, str]], feature_columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert rows to the feature matrix and binary unsupported labels."""
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
    """Negative log likelihood of the risk estimates."""
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return float(-(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)).mean())


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AUROC when defined."""
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float | None]:
    """Compute classification metrics after thresholding calibrated risk."""
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
    """Compute bin-based ECE."""
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


def _fit_platt_scaling(val_logits: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling on validation logits."""
    model = LogisticRegression(solver="lbfgs", random_state=42, max_iter=2000)
    model.fit(val_logits.reshape(-1, 1), y_val)
    return model


def _fit_isotonic_regression(raw_val_probs: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression on validation probabilities."""
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_val_probs, y_val)
    return model


def _collect_calibration_block(
    method_name: str,
    fit_payload: Dict[str, float | int],
    raw_metrics: Dict[str, Dict],
    calibrated_metrics: Dict[str, Dict],
) -> Dict:
    """Build the JSON block describing one selected calibration method."""
    return {
        "method": method_name,
        "fit_parameters": fit_payload,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
    }


def _build_curve_from_risk(y_true: np.ndarray, risk_scores: np.ndarray) -> List[Dict[str, float]]:
    """Sweep accepted-risk thresholds to build selective prediction curves."""
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
    """Choose the highest-accuracy validation point that meets minimum coverage."""
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
    """Draw the final reliability diagram for one split."""
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
    """Draw the selective risk and selective accuracy curves."""
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


def _build_frozen_bundle(
    *,
    source: str,
    feature_columns: Sequence[str],
    detector_coefficients: Dict[str, float],
    detector_intercept: float,
    calibration_method: str,
    calibration_fit_parameters: Dict[str, float | int],
    min_coverage: float,
    classification_threshold: float,
    frozen_threshold: float,
    selection_rule: str,
    validation_operating_point: Dict[str, float],
    test_operating_point: Dict[str, float],
    standardization_stats: Dict | None,
    tuned_report_path: Path,
    val_path: Path,
    test_path: Path,
) -> Dict:
    """Package the trained detector, calibrator, and abstention policy for reuse."""
    bundle = {
        "artifact_type": "frozen_detector_bundle",
        "source": source,
        "feature_columns": list(feature_columns),
        "detector": {
            "type": "logistic_regression",
            "coefficients": detector_coefficients,
            "intercept": detector_intercept,
        },
        "calibration": {
            "method": calibration_method,
            "fit_parameters": calibration_fit_parameters,
        },
        "abstention_policy": {
            "selection_rule": selection_rule,
            "minimum_coverage_constraint": min_coverage,
            "classification_threshold_for_metrics": classification_threshold,
            "frozen_risk_threshold": frozen_threshold,
            "answer_when_calibrated_risk_at_or_below": frozen_threshold,
            "abstain_when_calibrated_risk_above": frozen_threshold,
            "selected_validation_operating_point": validation_operating_point,
            "test_operating_point_at_frozen_threshold": test_operating_point,
        },
        "standardization": {
            "required": True,
            "feature_stats": None,
        },
        "provenance": {
            "tuned_report_path": str(tuned_report_path),
            "validation_split_path": str(val_path),
            "test_split_path": str(test_path),
        },
    }
    if standardization_stats is not None:
        bundle["standardization"]["feature_stats"] = standardization_stats.get("stats", standardization_stats)
        bundle["provenance"]["standardization_stats_path"] = standardization_stats.get("_loaded_from")
    return bundle


def main() -> None:
    """Run final calibration and abstention analysis and save all outputs."""
    parser = argparse.ArgumentParser(description="Calibrate a tuned detector and evaluate abstention.")
    parser.add_argument("--tuned-report", required=True, help="Path to tuned logreg report JSON")
    parser.add_argument("--val", required=True, help="Standardized validation CSV")
    parser.add_argument("--test", required=True, help="Standardized test CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for calibration outputs")
    parser.add_argument(
        "--standardization-stats",
        default=None,
        help="Optional JSON file with frozen feature standardization stats to embed in the saved bundle.",
    )
    parser.add_argument("--prefix", default="phantom_4000")
    parser.add_argument(
        "--source-name",
        default=None,
        help="Optional dataset name to save in the report and frozen bundle. If omitted, the script infers it from paths and prefix.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibration method to use for the final calibrated risk score. Default keeps the current PHANTOM behavior.",
    )
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
    tuned_report_path = Path(args.tuned_report)
    val_path = Path(args.val)
    test_path = Path(args.test)
    source_name = args.source_name or _infer_source_name(args.prefix, tuned_report_path, val_path, test_path)
    best_trial = tuned_report["best_trial"]
    coefficients = best_trial["coefficients"]
    intercept = float(best_trial["intercept"])
    feature_columns = list(coefficients.keys())
    selection_rule = "Choose the validation risk threshold that maximizes selective accuracy subject to the minimum coverage constraint."

    standardization_stats = None
    if args.standardization_stats:
        standardization_stats = _load_json(Path(args.standardization_stats))
        standardization_stats["_loaded_from"] = str(Path(args.standardization_stats))

    val_rows = _read_csv_rows(val_path)
    test_rows = _read_csv_rows(test_path)
    x_val, y_val = _rows_to_xy(val_rows, feature_columns)
    x_test, y_test = _rows_to_xy(test_rows, feature_columns)

    weights = np.array([coefficients[column] for column in feature_columns], dtype=np.float64)
    val_logits = x_val @ weights + intercept
    test_logits = x_test @ weights + intercept

    raw_val_probs = _sigmoid(val_logits)
    raw_test_probs = _sigmoid(test_logits)

    # PHANTOM currently keeps Platt as the final method. WikiQA can switch to
    # isotonic when the comparison stage shows it calibrates better.
    calibration_method_name = "Platt scaling" if args.calibration_method == "platt" else "Isotonic regression"
    calibration_fit_parameters: Dict[str, float | int]
    if args.calibration_method == "platt":
        platt_model = _fit_platt_scaling(val_logits, y_val)
        calibrated_val_probs = platt_model.predict_proba(val_logits.reshape(-1, 1))[:, 1]
        calibrated_test_probs = platt_model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
        calibration_fit_parameters = {
            "coef": float(platt_model.coef_[0][0]),
            "intercept": float(platt_model.intercept_[0]),
        }
    else:
        isotonic_model = _fit_isotonic_regression(raw_val_probs, y_val)
        calibrated_val_probs = np.clip(isotonic_model.predict(raw_val_probs), 0.0, 1.0)
        calibrated_test_probs = np.clip(isotonic_model.predict(raw_test_probs), 0.0, 1.0)
        calibration_fit_parameters = {
            "num_thresholds": int(len(isotonic_model.X_thresholds_)),
            "x_thresholds": [float(value) for value in isotonic_model.X_thresholds_.tolist()],
            "y_thresholds": [float(value) for value in isotonic_model.y_thresholds_.tolist()],
        }

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

    # The calibrated risk is the quantity used for abstention. Lower risk means
    # the system keeps the answer. Higher risk means the system abstains.
    val_curve = _build_curve_from_risk(y_val, calibrated_val_probs)
    test_curve = _build_curve_from_risk(y_test, calibrated_test_probs)
    operating_point = _select_operating_point(val_curve, min_coverage=args.min_coverage)
    frozen_threshold = float(operating_point["threshold"])

    selected_test_point = min(
        test_curve,
        key=lambda point: abs(point["threshold"] - frozen_threshold),
    )

    report = {
        "source": source_name,
        "feature_columns": feature_columns,
        "min_coverage_constraint": args.min_coverage,
        "classification_threshold_for_metrics": args.classification_threshold,
        "tuned_detector": {
            "coefficients": coefficients,
            "intercept": intercept,
        },
        "selected_calibration": _collect_calibration_block(
            calibration_method_name,
            calibration_fit_parameters,
            raw_metrics,
            calibrated_metrics,
        ),
        "abstention": {
            "selection_rule": selection_rule,
            "frozen_threshold": frozen_threshold,
            "selected_validation_operating_point": operating_point,
            "test_operating_point_at_frozen_threshold": selected_test_point,
            "validation_curve": val_curve,
            "test_curve": test_curve,
        },
    }
    if args.calibration_method == "platt":
        report["platt_scaling"] = report["selected_calibration"]
    else:
        report["isotonic_regression"] = report["selected_calibration"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.prefix}_calibration_abstention_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    frozen_bundle = _build_frozen_bundle(
        source=report["source"],
        feature_columns=feature_columns,
        detector_coefficients=coefficients,
        detector_intercept=intercept,
        calibration_method=calibration_method_name,
        calibration_fit_parameters=calibration_fit_parameters,
        min_coverage=args.min_coverage,
        classification_threshold=args.classification_threshold,
        frozen_threshold=frozen_threshold,
        selection_rule=selection_rule,
        validation_operating_point=operating_point,
        test_operating_point=selected_test_point,
        standardization_stats=standardization_stats,
        tuned_report_path=tuned_report_path,
        val_path=val_path,
        test_path=test_path,
    )
    bundle_path = output_dir / f"{args.prefix}_frozen_bundle.json"
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(frozen_bundle, handle, indent=2, ensure_ascii=False)

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
    print(f"Saved frozen detector bundle to: {bundle_path}")
    summary = {
        "calibration_method": calibration_method_name,
        "calibration_fit_parameters": calibration_fit_parameters,
        "validation_ece_before": raw_metrics["val"]["ece"],
        "validation_ece_after": calibrated_metrics["val"]["ece"],
        "validation_brier_before": raw_metrics["val"]["brier"],
        "validation_brier_after": calibrated_metrics["val"]["brier"],
        "test_ece_before": raw_metrics["test"]["ece"],
        "test_ece_after": calibrated_metrics["test"]["ece"],
        "test_brier_before": raw_metrics["test"]["brier"],
        "test_brier_after": calibrated_metrics["test"]["brier"],
        "frozen_threshold": frozen_threshold,
        "frozen_bundle_path": str(bundle_path),
        "selected_validation_operating_point": operating_point,
        "test_operating_point_at_frozen_threshold": selected_test_point,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
