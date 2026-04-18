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

BASELINE_FEATURE_SETS = {
    "token_only": ["mean_token_nll"],
    "self_consistency_only": ["self_consistency_disagreement"],
    "semantic_entropy_only": ["semantic_entropy"],
    "groundedness_only": ["groundedness_score"],
    "all_four_features": FEATURE_COLUMNS,
}


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


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float | None]:
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
    best_metrics = _classification_metrics(y_true, y_score, threshold=best_threshold)
    best_f1 = float(best_metrics["f1"])

    for threshold in candidate_thresholds:
        metrics = _classification_metrics(y_true, y_score, threshold=threshold)
        score = float(metrics["f1"])
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def _build_logreg(c_value: float, max_iter: int, class_weight: str) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="liblinear",
        random_state=42,
        class_weight=None if class_weight == "none" else class_weight,
    )


def _manual_weighted_scores(x: np.ndarray) -> np.ndarray:
    token = x[:, 0]
    self_consistency = x[:, 1]
    semantic = x[:, 2]
    groundedness = x[:, 3]
    raw_score = 0.10 * token + 0.20 * self_consistency + 0.30 * semantic - 0.40 * groundedness
    return 1.0 / (1.0 + np.exp(-raw_score))


def _tune_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: Sequence[str],
    c_values: Sequence[float],
    class_weights: Sequence[str],
    max_iter: int,
) -> Dict:
    best_trial = None
    best_objective = float("-inf")
    all_trials = []

    for c_value in c_values:
        for class_weight in class_weights:
            model = _build_logreg(c_value, max_iter, class_weight)
            model.fit(x_train, y_train)

            train_scores = model.predict_proba(x_train)[:, 1]
            val_scores = model.predict_proba(x_val)[:, 1]
            test_scores = model.predict_proba(x_test)[:, 1]

            threshold, val_metrics = _find_best_threshold_for_f1(y_val, val_scores)
            trial = {
                "C": c_value,
                "class_weight": class_weight,
                "selected_threshold": threshold,
                "feature_columns": list(feature_columns),
                "coefficients": {
                    feature_columns[i]: float(model.coef_[0][i]) for i in range(len(feature_columns))
                },
                "intercept": float(model.intercept_[0]),
                "train_metrics": _classification_metrics(y_train, train_scores, threshold),
                "val_metrics": val_metrics,
                "test_metrics": _classification_metrics(y_test, test_scores, threshold),
            }
            all_trials.append(trial)

            objective = float(val_metrics["f1"])
            if objective > best_objective:
                best_objective = objective
                best_trial = trial

    return {
        "selection_metric": "validation_f1",
        "num_trials": len(all_trials),
        "best_trial": best_trial,
        "all_trials": all_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run practical detector baselines on standardized feature splits.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="phantom_4000")
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--class-weight-grid", default="none,balanced")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_rows = _read_csv_rows(Path(args.train))
    val_rows = _read_csv_rows(Path(args.val))
    test_rows = _read_csv_rows(Path(args.test))

    c_values = [float(value.strip()) for value in args.c_grid.split(",") if value.strip()]
    class_weights = [value.strip() for value in args.class_weight_grid.split(",") if value.strip()]

    baselines: Dict[str, Dict] = {}
    for baseline_name, feature_columns in BASELINE_FEATURE_SETS.items():
        x_train, y_train = _rows_to_xy(train_rows, feature_columns)
        x_val, y_val = _rows_to_xy(val_rows, feature_columns)
        x_test, y_test = _rows_to_xy(test_rows, feature_columns)
        baselines[baseline_name] = _tune_logreg(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            feature_columns=feature_columns,
            c_values=c_values,
            class_weights=class_weights,
            max_iter=args.max_iter,
        )

    x_train_all, y_train_all = _rows_to_xy(train_rows, FEATURE_COLUMNS)
    x_val_all, y_val_all = _rows_to_xy(val_rows, FEATURE_COLUMNS)
    x_test_all, y_test_all = _rows_to_xy(test_rows, FEATURE_COLUMNS)

    manual_train = _manual_weighted_scores(x_train_all)
    manual_val = _manual_weighted_scores(x_val_all)
    manual_test = _manual_weighted_scores(x_test_all)
    manual_threshold, manual_val_metrics = _find_best_threshold_for_f1(y_val_all, manual_val)

    baselines["manual_weighted"] = {
        "selection_metric": "validation_f1",
        "best_trial": {
            "selected_threshold": manual_threshold,
            "weights": {
                "mean_token_nll": 0.10,
                "self_consistency_disagreement": 0.20,
                "semantic_entropy": 0.30,
                "groundedness_score": -0.40,
            },
            "train_metrics": _classification_metrics(y_train_all, manual_train, manual_threshold),
            "val_metrics": manual_val_metrics,
            "test_metrics": _classification_metrics(y_test_all, manual_test, manual_threshold),
        },
    }

    rng = np.random.default_rng(args.seed)
    random_train = rng.random(len(y_train_all))
    random_val = rng.random(len(y_val_all))
    random_test = rng.random(len(y_test_all))
    random_threshold, random_val_metrics = _find_best_threshold_for_f1(y_val_all, random_val)
    baselines["random_score"] = {
        "selection_metric": "validation_f1",
        "best_trial": {
            "selected_threshold": random_threshold,
            "train_metrics": _classification_metrics(y_train_all, random_train, random_threshold),
            "val_metrics": random_val_metrics,
            "test_metrics": _classification_metrics(y_test_all, random_test, random_threshold),
        },
    }

    report = {
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "baselines": baselines,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.prefix}_baseline_report.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Saved baseline report to: {output_path}")
    summary = {
        name: baseline["best_trial"]["test_metrics"]
        for name, baseline in baselines.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
