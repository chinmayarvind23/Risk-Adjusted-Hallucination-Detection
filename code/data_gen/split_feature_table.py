from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label_histogram(rows: List[Dict[str, str]], label_key: str = "judge_binary_label") -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for row in rows:
        value = row.get(label_key)
        if value is None or value == "":
            continue
        histogram[value] = histogram.get(value, 0) + 1
    return histogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a flat feature-table CSV into train/val/test CSVs.")
    parser.add_argument("--input", required=True, help="Input feature table CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for split CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="phantom")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows = _read_csv(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    random.Random(args.seed).shuffle(rows)

    total = len(rows)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    fieldnames = list(rows[0].keys())

    train_path = output_dir / f"{args.prefix}_train.csv"
    val_path = output_dir / f"{args.prefix}_val.csv"
    test_path = output_dir / f"{args.prefix}_test.csv"
    summary_path = output_dir / f"{args.prefix}_split_summary.json"

    _write_csv(train_path, train_rows, fieldnames)
    _write_csv(val_path, val_rows, fieldnames)
    _write_csv(test_path, test_rows, fieldnames)

    summary = {
        "input": str(input_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total_rows": total,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_label_histogram": _label_histogram(train_rows),
        "val_label_histogram": _label_histogram(val_rows),
        "test_label_histogram": _label_histogram(test_rows),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved train split to: {train_path}")
    print(f"Saved val split to: {val_path}")
    print(f"Saved test split to: {test_path}")
    print(f"Saved split summary to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
