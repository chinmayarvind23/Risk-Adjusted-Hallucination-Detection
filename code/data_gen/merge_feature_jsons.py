from __future__ import annotations

"""Merge multiple generated feature JSON files and compute a simple summary.

This is helpful when feature generation was run in parts and the detector needs
one combined file.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    """Read one JSON file that stores a list of example records."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Average optional floats while skipping missing values."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _label_histogram(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    """Count label values for one field across all records."""
    histogram: Dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    return histogram


def _judge_histogram(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count binary judge labels from the nested judge block."""
    histogram: Dict[str, int] = {}
    for record in records:
        judge = record.get("judge_label") or {}
        if "binary_label" not in judge:
            continue
        key = str(judge["binary_label"])
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def summarize(records: Sequence[Dict[str, Any]], source_files: Sequence[Path]) -> Dict[str, Any]:
    """Build a compact dataset-level summary of the merged feature file."""
    summary: Dict[str, Any] = {
        "num_examples": len(records),
        "source_files": [str(path) for path in source_files],
    }

    summary["mean_token_nll"] = _mean(
        [
            (record.get("token_uncertainty") or {}).get("mean_token_nll")
            for record in records
        ]
    )
    summary["mean_self_consistency_disagreement"] = _mean(
        [
            (record.get("self_consistency") or {}).get("disagreement_score")
            for record in records
        ]
    )
    summary["mean_semantic_entropy"] = _mean(
        [
            (record.get("semantic_entropy") or {}).get("semantic_entropy")
            for record in records
        ]
    )
    summary["mean_groundedness_score"] = _mean(
        [
            (record.get("evidence_consistency") or {}).get("groundedness_score")
            for record in records
        ]
    )

    gt_histogram = _label_histogram(records, "ground_truth_label")
    if gt_histogram:
        summary["ground_truth_label_histogram"] = gt_histogram

    judge_histogram = _judge_histogram(records)
    if judge_histogram:
        summary["judge_label_histogram"] = judge_histogram

    return summary


def main() -> None:
    """CLI entry point for merging partial JSON feature files."""
    parser = argparse.ArgumentParser(description="Merge per-split feature JSON files into one JSON file.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input feature JSON files to merge in order.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional merged summary JSON path. Defaults to <output stem>_summary.json.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )

    merged: List[Dict[str, Any]] = []
    for path in input_paths:
        rows = _load_json_list(path)
        print(f"Loaded {len(rows)} rows from {path}")
        merged.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)

    summary = summarize(merged, input_paths)
    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved merged JSON to: {output_path}")
    print(f"Saved merged summary to: {summary_output}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
