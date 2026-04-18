from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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


def _rows_to_xy(rows: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [float(row[column]) for column in FEATURE_COLUMNS]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.array([int(row["judge_binary_label"]) for row in rows], dtype=np.int64)
    return x, y


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": _safe_roc_auc(y_true, y_score),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_at_threshold": float(y_pred.mean()),
    }


def _find_best_threshold_for_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float | None]]:
    candidate_thresholds = sorted(set(float(score) for score in y_score.tolist()))
    candidate_thresholds = [0.0] + candidate_thresholds + [1.0]

    best_threshold = 0.5
    best_metrics = _classification_metrics(y_true, y_score, threshold=0.5)
    best_f1 = float(best_metrics["f1"])

    for threshold in candidate_thresholds:
        metrics = _classification_metrics(y_true, y_score, threshold=threshold)
        current_f1 = float(metrics["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def _manual_weighted_scores(x: np.ndarray) -> np.ndarray:
    """
    Ablation baseline only.

    After z-score standardization:
    - higher token/self/semantic => more hallucination risk
    - higher groundedness => less hallucination risk

    Manual priority requested by user:
    evidence consistency > self-consistency > semantic > token
    """
    token = x[:, 0]
    self_consistency = x[:, 1]
    semantic = x[:, 2]
    groundedness = x[:, 3]

    raw_score = (
        0.10 * token
        + 0.20 * self_consistency
        + 0.30 * semantic
        - 0.40 * groundedness
    )

    # Logistic squashing so the ablation produces a probability-like risk score.
    return 1.0 / (1.0 + np.exp(-raw_score))


def _build_logreg(C: float, max_iter: int, class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="liblinear",
        random_state=42,
        class_weight=None if class_weight == "none" else class_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic regression detector on standardized feature splits.")
    parser.add_argument("--train", required=True, help="Standardized train CSV")
    parser.add_argument("--val", required=True, help="Standardized validation CSV")
    parser.add_argument("--test", required=True, help="Standardized test CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for model report outputs")
    parser.add_argument("--prefix", default="phantom_4000")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength for logistic regression")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tune", action="store_true", help="Run a practical hyperparameter sweep and keep the best model.")
    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated C values to try when --tune is enabled.",
    )
    parser.add_argument(
        "--class-weight-grid",
        default="none,balanced",
        help="Comma-separated class_weight settings to try when --tune is enabled.",
    )
    args = parser.parse_args()

    train_rows = _read_csv_rows(Path(args.train))
    val_rows = _read_csv_rows(Path(args.val))
    test_rows = _read_csv_rows(Path(args.test))

    x_train, y_train = _rows_to_xy(train_rows)
    x_val, y_val = _rows_to_xy(val_rows)
    x_test, y_test = _rows_to_xy(test_rows)

    model = _build_logreg(C=args.c, max_iter=args.max_iter, class_weight="none")
    model.fit(x_train, y_train)

    train_scores = model.predict_proba(x_train)[:, 1]
    val_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    train_metrics = _classification_metrics(y_train, train_scores)
    val_metrics = _classification_metrics(y_val, val_scores)
    test_metrics = _classification_metrics(y_test, test_scores)

    manual_train_scores = _manual_weighted_scores(x_train)
    manual_val_scores = _manual_weighted_scores(x_val)
    manual_test_scores = _manual_weighted_scores(x_test)

    manual_train_metrics = _classification_metrics(y_train, manual_train_scores)
    manual_val_metrics = _classification_metrics(y_val, manual_val_scores)
    manual_test_metrics = _classification_metrics(y_test, manual_test_scores)

    tuned_report = None
    if args.tune:
        c_values = [float(value.strip()) for value in args.c_grid.split(",") if value.strip()]
        class_weight_values = [value.strip() for value in args.class_weight_grid.split(",") if value.strip()]

        tuning_trials = []
        best_trial = None
        best_objective = float("-inf")

        for c_value in c_values:
            for class_weight_value in class_weight_values:
                candidate_model = _build_logreg(
                    C=c_value,
                    max_iter=args.max_iter,
                    class_weight=class_weight_value,
                )
                candidate_model.fit(x_train, y_train)

                train_candidate_scores = candidate_model.predict_proba(x_train)[:, 1]
                val_candidate_scores = candidate_model.predict_proba(x_val)[:, 1]
                test_candidate_scores = candidate_model.predict_proba(x_test)[:, 1]

                best_threshold, tuned_val_metrics = _find_best_threshold_for_f1(y_val, val_candidate_scores)
                tuned_train_metrics = _classification_metrics(y_train, train_candidate_scores, threshold=best_threshold)
                tuned_test_metrics = _classification_metrics(y_test, test_candidate_scores, threshold=best_threshold)

                trial = {
                    "C": c_value,
                    "class_weight": class_weight_value,
                    "selected_threshold": best_threshold,
                    "coefficients": {
                        FEATURE_COLUMNS[i]: float(candidate_model.coef_[0][i]) for i in range(len(FEATURE_COLUMNS))
                    },
                    "intercept": float(candidate_model.intercept_[0]),
                    "train_metrics": tuned_train_metrics,
                    "val_metrics": tuned_val_metrics,
                    "test_metrics": tuned_test_metrics,
                }
                tuning_trials.append(trial)

                objective = float(tuned_val_metrics["f1"])
                if objective > best_objective:
                    best_objective = objective
                    best_trial = trial

        tuned_report = {
            "selection_metric": "validation_f1",
            "c_grid": c_values,
            "class_weight_grid": class_weight_values,
            "num_trials": len(tuning_trials),
            "best_trial": best_trial,
            "all_trials": tuning_trials,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "logistic_regression": {
            "hyperparameters": {
                "C": args.c,
                "max_iter": args.max_iter,
                "solver": "liblinear",
            },
            "intercept": float(model.intercept_[0]),
            "coefficients": {
                FEATURE_COLUMNS[i]: float(model.coef_[0][i]) for i in range(len(FEATURE_COLUMNS))
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        "manual_weighted_ablation": {
            "weights": {
                "mean_token_nll": 0.10,
                "self_consistency_disagreement": 0.20,
                "semantic_entropy": 0.30,
                "groundedness_score": -0.40,
            },
            "train_metrics": manual_train_metrics,
            "val_metrics": manual_val_metrics,
            "test_metrics": manual_test_metrics,
        },
    }
    if tuned_report is not None:
        report["hyperparameter_tuning"] = tuned_report

    report_path = output_dir / f"{args.prefix}_logreg_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Saved detector report to: {report_path}")
    print(json.dumps(report["logistic_regression"], indent=2))
    print(json.dumps({"manual_weighted_ablation": report["manual_weighted_ablation"]}, indent=2))
    if tuned_report is not None:
        tuned_path = output_dir / f"{args.prefix}_logreg_tuned_report.json"
        with tuned_path.open("w", encoding="utf-8") as handle:
            json.dump(tuned_report, handle, indent=2, ensure_ascii=False)
        print(f"Saved tuned detector report to: {tuned_path}")
        print(json.dumps({"best_trial": tuned_report["best_trial"]}, indent=2))


if __name__ == "__main__":
    main()
