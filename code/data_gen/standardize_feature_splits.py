from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


FEATURE_COLUMNS = [
    "mean_token_nll",
    "self_consistency_disagreement",
    "semantic_entropy",
    "groundedness_score",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str) -> float:
    return float(value)


def _fit_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        values = [_to_float(row[column]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def _transform_rows(rows: List[Dict[str, str]], stats: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    transformed: List[Dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        for column in FEATURE_COLUMNS:
            value = _to_float(row[column])
            mean = stats[column]["mean"]
            std = stats[column]["std"]
            z_value = (value - mean) / std
            new_row[column] = f"{z_value:.12g}"
        transformed.append(new_row)
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit z-score stats on train split and transform train/val/test CSVs.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="phantom_4000")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    output_dir = Path(args.output_dir)

    train_rows = _read_csv(train_path)
    val_rows = _read_csv(val_path)
    test_rows = _read_csv(test_path)

    if not train_rows:
        raise ValueError("Train split is empty")

    stats = _fit_stats(train_rows)
    train_z = _transform_rows(train_rows, stats)
    val_z = _transform_rows(val_rows, stats)
    test_z = _transform_rows(test_rows, stats)

    fieldnames = list(train_rows[0].keys())
    train_out = output_dir / f"{args.prefix}_train_standardized.csv"
    val_out = output_dir / f"{args.prefix}_val_standardized.csv"
    test_out = output_dir / f"{args.prefix}_test_standardized.csv"
    stats_out = output_dir / f"{args.prefix}_standardization_stats.json"

    _write_csv(train_out, train_z, fieldnames)
    _write_csv(val_out, val_z, fieldnames)
    _write_csv(test_out, test_z, fieldnames)

    with stats_out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_rows": len(test_rows),
                "stats": stats,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved standardized train split to: {train_out}")
    print(f"Saved standardized val split to: {val_out}")
    print(f"Saved standardized test split to: {test_out}")
    print(f"Saved standardization stats to: {stats_out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
