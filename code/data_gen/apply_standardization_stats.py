from __future__ import annotations

"""Apply frozen training-set normalization stats to new flat feature tables.

This script exists for transfer and cross-regime evaluation. It intentionally
does not fit new mean and standard deviation values. Instead it reuses the
statistics from a saved training regime so new data enters the detector in the
same feature space.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


FEATURE_COLUMNS = [
    "mean_token_nll",
    "self_consistency_disagreement",
    "semantic_entropy",
    "groundedness_score",
]


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a flat feature CSV while allowing long text fields."""
    _set_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """Load normalization stats from either a stats file or a frozen bundle."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "stats" in payload:
        return payload["stats"]
    if "standardization" in payload and payload["standardization"].get("feature_stats"):
        return payload["standardization"]["feature_stats"]

    raise ValueError(f"Could not find feature stats in {path}")


def _transform_rows(rows: List[Dict[str, str]], stats: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    """Apply per-column z-score normalization using already frozen statistics."""
    transformed: List[Dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        for column in FEATURE_COLUMNS:
            value = float(row[column])
            mean = float(stats[column]["mean"])
            std = float(stats[column]["std"])
            if std == 0.0:
                std = 1.0
            z_value = (value - mean) / std
            new_row[column] = f"{z_value:.12g}"
        transformed.append(new_row)
    return transformed


def main() -> None:
    """CLI entry point for applying frozen normalization to one or more CSV files."""
    parser = argparse.ArgumentParser(
        description="Apply frozen feature standardization stats to one or more flat feature-table CSV files."
    )
    parser.add_argument(
        "--stats-json",
        required=True,
        help="JSON file containing frozen feature standardization stats. Can be a standardization_stats file or a frozen bundle.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input CSV files to standardize with the same frozen stats.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where standardized CSV files will be written.",
    )
    parser.add_argument(
        "--suffix",
        default="_standardized",
        help="Suffix added before the .csv extension for each output file.",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats_json)
    output_dir = Path(args.output_dir)
    stats = _load_stats(stats_path)

    for input_name in args.inputs:
        input_path = Path(input_name)
        rows = _read_csv(input_path)
        if not rows:
            raise ValueError(f"No rows found in {input_path}")

        transformed = _transform_rows(rows, stats)
        output_name = f"{input_path.stem}{args.suffix}.csv"
        output_path = output_dir / output_name
        _write_csv(output_path, transformed, list(rows[0].keys()))
        print(f"Saved standardized CSV to: {output_path}")


if __name__ == "__main__":
    main()
