from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


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
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
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

    return {"temperature": fine_best_temp}


def _collect_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "ece": _expected_calibration_error(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "nll": _nll(y_true, probs),
    }


def _reliability_points(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[List[float], List[float]]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = []
    bin_acc = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        bin_conf.append(float(probs[mask].mean()))
        bin_acc.append(float(y_true[mask].mean()))
    return bin_conf, bin_acc


def _plot_reliability_comparison(
    y_true: np.ndarray,
    method_probs: Dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    colors = {
        "raw": "#666666",
        "temperature_scaling": "#0f4c81",
        "platt_scaling": "#0a9396",
        "isotonic_regression": "#bb3e03",
    }
    labels = {
        "raw": "Raw",
        "temperature_scaling": "Temperature",
        "platt_scaling": "Platt",
        "isotonic_regression": "Isotonic",
    }

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    for name, probs in method_probs.items():
        conf, acc = _reliability_points(y_true, probs)
        ax.plot(conf, acc, marker="o", linewidth=2, color=colors[name], label=labels[name])

    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Observed Unsupported Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibration methods on validation only and evaluate on test.")
    parser.add_argument("--tuned-report", required=True, help="Path to tuned logreg report JSON")
    parser.add_argument("--val", required=True, help="Standardized validation CSV")
    parser.add_argument("--test", required=True, help="Standardized test CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison outputs")
    parser.add_argument("--prefix", default="phantom_4000")
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
    temp_val_probs = _sigmoid(val_logits / temperature)
    temp_test_probs = _sigmoid(test_logits / temperature)

    platt = LogisticRegression(solver="lbfgs", random_state=42, max_iter=2000)
    platt.fit(val_logits.reshape(-1, 1), y_val)
    platt_val_probs = platt.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    platt_test_probs = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_val_probs, y_val)
    iso_val_probs = np.clip(isotonic.predict(raw_val_probs), 0.0, 1.0)
    iso_test_probs = np.clip(isotonic.predict(raw_test_probs), 0.0, 1.0)

    results = {
        "validation_fit_only": True,
        "methods": {
            "raw": {
                "fit": {},
                "val_metrics": _collect_metrics(y_val, raw_val_probs),
                "test_metrics": _collect_metrics(y_test, raw_test_probs),
            },
            "temperature_scaling": {
                "fit": {"temperature": temperature},
                "val_metrics": _collect_metrics(y_val, temp_val_probs),
                "test_metrics": _collect_metrics(y_test, temp_test_probs),
            },
            "platt_scaling": {
                "fit": {
                    "coef": float(platt.coef_[0][0]),
                    "intercept": float(platt.intercept_[0]),
                },
                "val_metrics": _collect_metrics(y_val, platt_val_probs),
                "test_metrics": _collect_metrics(y_test, platt_test_probs),
            },
            "isotonic_regression": {
                "fit": {"num_thresholds": int(len(isotonic.X_thresholds_))},
                "val_metrics": _collect_metrics(y_val, iso_val_probs),
                "test_metrics": _collect_metrics(y_test, iso_test_probs),
            },
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{args.prefix}_calibration_method_comparison.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    _plot_reliability_comparison(
        y_val,
        {
            "raw": raw_val_probs,
            "temperature_scaling": temp_val_probs,
            "platt_scaling": platt_val_probs,
            "isotonic_regression": iso_val_probs,
        },
        output_dir / f"{args.prefix}_reliability_comparison_val.png",
        "Reliability Diagram Comparison on Validation Set",
    )

    _plot_reliability_comparison(
        y_test,
        {
            "raw": raw_test_probs,
            "temperature_scaling": temp_test_probs,
            "platt_scaling": platt_test_probs,
            "isotonic_regression": iso_test_probs,
        },
        output_dir / f"{args.prefix}_reliability_comparison_test.png",
        "Reliability Diagram Comparison on Test Set",
    )

    summary = {
        method_name: payload["test_metrics"]
        for method_name, payload in results["methods"].items()
    }
    print(f"Saved calibration comparison report to: {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
