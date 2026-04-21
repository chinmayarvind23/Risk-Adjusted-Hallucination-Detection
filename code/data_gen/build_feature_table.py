from __future__ import annotations

"""Flatten merged feature JSON into a row-wise CSV table for detector training.

The generator stores nested JSON because it is convenient for feature
computation and inspection. The detector stage needs one row per example with
one column per feature, so this file performs that flattening step.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_get(mapping: Optional[Dict[str, Any]], key: str) -> Any:
    """Return a nested value safely when optional feature blocks are missing."""
    if not mapping:
        return None
    return mapping.get(key)


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one nested generation record into one flat detector row."""
    token_uncertainty = record.get("token_uncertainty") or {}
    self_consistency = record.get("self_consistency") or {}
    semantic_entropy = record.get("semantic_entropy") or {}
    evidence_consistency = record.get("evidence_consistency") or {}
    judge_label = record.get("judge_label") or {}

    row = {
        "example_id": record.get("example_id"),
        "dataset_name": record.get("dataset_name"),
        "question": record.get("question"),
        "context": record.get("context"),
        "served_answer": record.get("served_answer"),
        "ground_truth_answer": record.get("ground_truth_answer"),
        "ground_truth_label": record.get("ground_truth_label"),
        "judge_binary_label": _safe_get(judge_label, "binary_label"),
        "judge_label": _safe_get(judge_label, "label"),
        "mean_token_nll": _safe_get(token_uncertainty, "mean_token_nll"),
        "mean_token_logprob": _safe_get(token_uncertainty, "mean_token_logprob"),
        "sum_token_logprob": _safe_get(token_uncertainty, "sum_token_logprob"),
        "num_scored_tokens": _safe_get(token_uncertainty, "num_scored_tokens"),
        "mean_next_token_entropy": _safe_get(token_uncertainty, "mean_next_token_entropy"),
        "self_consistency_score": _safe_get(self_consistency, "base_consistency_score"),
        "self_consistency_disagreement": _safe_get(self_consistency, "disagreement_score"),
        "semantic_entropy": _safe_get(semantic_entropy, "semantic_entropy"),
        "normalized_semantic_entropy": _safe_get(semantic_entropy, "normalized_semantic_entropy"),
        "num_semantic_clusters": _safe_get(semantic_entropy, "num_semantic_clusters"),
        "groundedness_score": _safe_get(evidence_consistency, "groundedness_score"),
        "discrete_groundedness_score": _safe_get(evidence_consistency, "discrete_groundedness_score"),
        "entailed_fraction": _safe_get(evidence_consistency, "entailed_fraction"),
        "contradicted_fraction": _safe_get(evidence_consistency, "contradicted_fraction"),
        "neutral_fraction": _safe_get(evidence_consistency, "neutral_fraction"),
        "mean_entailment": _safe_get(evidence_consistency, "mean_entailment"),
        "mean_contradiction": _safe_get(evidence_consistency, "mean_contradiction"),
        "mean_neutral": _safe_get(evidence_consistency, "mean_neutral"),
    }
    return row


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load the merged JSON list produced after feature generation."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _write_csv(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Write the flat feature table to CSV for later split and training steps."""
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Optional JSONL output for workflows that prefer line-delimited records."""
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    """CLI entry point for building a flat feature table from merged JSON."""
    parser = argparse.ArgumentParser(description="Build a flat feature table from merged generation-feature JSON.")
    parser.add_argument("--input", required=True, help="Merged feature JSON file.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional output JSONL path for the same flattened feature table.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None

    records = _load_records(input_path)
    rows = [_flatten_record(record) for record in records]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_csv)
    print(f"Saved feature table CSV to: {output_csv}")
    print(f"Rows: {len(rows)}")

    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(rows, output_jsonl)
        print(f"Saved feature table JSONL to: {output_jsonl}")


if __name__ == "__main__":
    main()
