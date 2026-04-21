from __future__ import annotations

"""
Compute deeper diagnostics for the two standalone runs and both transfer runs.

This script is meant to answer a few research-style questions that are not
covered by the main training reports alone:

1. Which feature-label relationships stay stable across datasets?
2. Which feature directions flip across datasets?
3. Does transfer fail mostly in ranking, calibration, or thresholding?
4. How many unsupported examples remain below the frozen abstention threshold?

It writes one JSON file that can be cited directly in the final results summary.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "mean_token_nll",
    "self_consistency_disagreement",
    "semantic_entropy",
    "groundedness_score",
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _dataset_feature_summary(name: str, csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    overall = {
        feature: {
            "mean": float(df[feature].mean()),
            "std": float(df[feature].std()),
        }
        for feature in FEATURE_COLUMNS
    }

    by_label = {}
    for label in [0, 1]:
        subset = df[df["judge_binary_label"] == label]
        by_label[str(label)] = {
            feature: {
                "mean": float(subset[feature].mean()),
                "std": float(subset[feature].std()),
            }
            for feature in FEATURE_COLUMNS
        }

    unsupported_minus_supported = {
        feature: float(
            by_label["1"][feature]["mean"] - by_label["0"][feature]["mean"]
        )
        for feature in FEATURE_COLUMNS
    }

    return {
        "dataset": name,
        "rows": int(len(df)),
        "unsupported_rate": float(df["judge_binary_label"].mean()),
        "overall": overall,
        "by_label": by_label,
        "unsupported_minus_supported": unsupported_minus_supported,
    }


def _compare_feature_directions(source_summary: Dict, target_summary: Dict) -> Dict:
    comparison = {}
    for feature in FEATURE_COLUMNS:
        source_delta = source_summary["unsupported_minus_supported"][feature]
        target_delta = target_summary["unsupported_minus_supported"][feature]
        same_direction = (source_delta == 0.0 or target_delta == 0.0) or (
            np.sign(source_delta) == np.sign(target_delta)
        )
        comparison[feature] = {
            "source_delta": source_delta,
            "target_delta": target_delta,
            "same_direction": bool(same_direction),
            "direction_flip": bool(not same_direction),
        }
    return comparison


def _standalone_summary(
    name: str,
    tuned_report_path: Path,
    calibration_report_path: Path,
) -> Dict:
    tuned = _load_json(tuned_report_path)
    calibration = _load_json(calibration_report_path)

    if name.lower() == "phantom":
        calibrated_block = calibration["platt_scaling"]["calibrated_metrics"]["test"]
        full_coverage_accuracy = 1.0 - calibration["abstention"]["test_curve"][-1]["selective_risk"]
    else:
        calibrated_block = calibration["selected_calibration"]["calibrated_metrics"]["test"]
        full_coverage_accuracy = 1.0 - calibration["abstention"]["test_curve"][-1]["selective_risk"]

    selective_accuracy = calibration["abstention"]["test_operating_point_at_frozen_threshold"]["selective_accuracy"]
    abstention_gain = selective_accuracy - full_coverage_accuracy

    return {
        "dataset": name,
        "detector_threshold": float(tuned["best_trial"]["selected_threshold"]),
        "detector_test_metrics": tuned["best_trial"]["test_metrics"],
        "calibration_method": calibration["selected_calibration"]["method"] if "selected_calibration" in calibration else calibration["platt_scaling"]["method"],
        "calibrated_test_metrics": calibrated_block,
        "full_coverage_accuracy_from_abstention_curve": float(full_coverage_accuracy),
        "frozen_threshold_test_operating_point": calibration["abstention"]["test_operating_point_at_frozen_threshold"],
        "abstention_accuracy_gain": float(abstention_gain),
    }


def _apply_bundle(bundle: Dict, standardized_csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(standardized_csv_path)
    weights = np.array(
        [bundle["detector"]["coefficients"][column] for column in bundle["feature_columns"]],
        dtype=np.float64,
    )
    x = df[bundle["feature_columns"]].to_numpy(dtype=np.float64)
    logits = x @ weights + float(bundle["detector"]["intercept"])
    raw_probs = _sigmoid(logits)

    calibration = bundle["calibration"]
    method = calibration["method"].strip().lower()
    if method == "platt scaling":
        fit = calibration["fit_parameters"]
        calibrated_probs = _sigmoid(
            float(fit["coef"]) * logits + float(fit["intercept"])
        )
    elif method == "isotonic regression":
        fit = calibration["fit_parameters"]
        calibrated_probs = np.interp(
            raw_probs,
            np.array(fit["x_thresholds"], dtype=np.float64),
            np.array(fit["y_thresholds"], dtype=np.float64),
            left=float(fit["y_thresholds"][0]),
            right=float(fit["y_thresholds"][-1]),
        )
        calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported bundle calibration method: {calibration['method']}")

    return df, raw_probs, calibrated_probs


def _transfer_failure_breakdown(
    transfer_report_path: Path,
    bundle_path: Path,
    standardized_target_csv_path: Path,
) -> Dict:
    transfer = _load_json(transfer_report_path)
    bundle = _load_json(bundle_path)
    df, raw_probs, calibrated_probs = _apply_bundle(bundle, standardized_target_csv_path)

    y = df["judge_binary_label"].to_numpy(dtype=np.int64)
    frozen_threshold = float(bundle["abstention_policy"]["frozen_risk_threshold"])

    unsupported_mask = y == 1
    supported_mask = y == 0
    kept_mask = calibrated_probs <= frozen_threshold
    confident_wrong_mask = unsupported_mask & kept_mask

    confident_wrong_rate_within_unsupported = (
        float(confident_wrong_mask.sum() / unsupported_mask.sum())
        if unsupported_mask.sum() > 0
        else 0.0
    )

    confident_wrong_feature_means = {}
    if confident_wrong_mask.any():
        subset = df.loc[confident_wrong_mask, FEATURE_COLUMNS]
        confident_wrong_feature_means = {
            feature: float(subset[feature].mean()) for feature in FEATURE_COLUMNS
        }

    return {
        "source_name": transfer["source_name"],
        "target_name": transfer["target_name"],
        "raw_test_metrics": transfer["raw_metrics"],
        "calibrated_test_metrics": transfer["calibrated_metrics"],
        "frozen_target_operating_point": transfer["transfer_abstention"]["target_operating_point_at_frozen_threshold"],
        "raw_to_calibrated_metric_change": {
            "auroc": float(transfer["calibrated_metrics"]["classification_metrics"]["auroc"] - transfer["raw_metrics"]["classification_metrics"]["auroc"]),
            "ece": float(transfer["calibrated_metrics"]["ece"] - transfer["raw_metrics"]["ece"]),
            "brier": float(transfer["calibrated_metrics"]["brier"] - transfer["raw_metrics"]["brier"]),
            "f1": float(transfer["calibrated_metrics"]["classification_metrics"]["f1"] - transfer["raw_metrics"]["classification_metrics"]["f1"]),
        },
        "unsupported_examples": int(unsupported_mask.sum()),
        "supported_examples": int(supported_mask.sum()),
        "unsupported_kept_below_frozen_threshold": int(confident_wrong_mask.sum()),
        "unsupported_kept_rate_within_unsupported": float(confident_wrong_rate_within_unsupported),
        "confident_wrong_feature_means": confident_wrong_feature_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deeper diagnostics for standalone and transfer runs.")
    parser.add_argument("--phantom-raw", required=True)
    parser.add_argument("--wikiqa-raw", required=True)
    parser.add_argument("--phantom-tuned-report", required=True)
    parser.add_argument("--wikiqa-tuned-report", required=True)
    parser.add_argument("--phantom-calibration-report", required=True)
    parser.add_argument("--wikiqa-calibration-report", required=True)
    parser.add_argument("--phantom-bundle", required=True)
    parser.add_argument("--wikiqa-bundle", required=True)
    parser.add_argument("--phantom-to-wikiqa-report", required=True)
    parser.add_argument("--wikiqa-to-phantom-report", required=True)
    parser.add_argument("--wikiqa-transfer-standardized", required=True)
    parser.add_argument("--phantom-transfer-standardized", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    phantom_summary = _dataset_feature_summary("PHANTOM", Path(args.phantom_raw))
    wikiqa_summary = _dataset_feature_summary("WikiQA", Path(args.wikiqa_raw))

    diagnostics = {
        "feature_distribution_shift": {
            "phantom": phantom_summary,
            "wikiqa": wikiqa_summary,
            "direction_comparison": _compare_feature_directions(phantom_summary, wikiqa_summary),
        },
        "standalone_runs": {
            "phantom": _standalone_summary(
                "PHANTOM",
                Path(args.phantom_tuned_report),
                Path(args.phantom_calibration_report),
            ),
            "wikiqa": _standalone_summary(
                "WikiQA",
                Path(args.wikiqa_tuned_report),
                Path(args.wikiqa_calibration_report),
            ),
        },
        "transfer_runs": {
            "phantom_to_wikiqa": _transfer_failure_breakdown(
                Path(args.phantom_to_wikiqa_report),
                Path(args.phantom_bundle),
                Path(args.wikiqa_transfer_standardized),
            ),
            "wikiqa_to_phantom": _transfer_failure_breakdown(
                Path(args.wikiqa_to_phantom_report),
                Path(args.wikiqa_bundle),
                Path(args.phantom_transfer_standardized),
            ),
        },
        "coefficient_comparison": {
            "phantom": _load_json(Path(args.phantom_bundle))["detector"]["coefficients"],
            "wikiqa": _load_json(Path(args.wikiqa_bundle))["detector"]["coefficients"],
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    print(f"Saved transfer diagnostics to: {output_path}")


if __name__ == "__main__":
    main()
