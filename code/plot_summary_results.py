from __future__ import annotations

"""Plot a small side by side summary for PHANTOM and WikiQA runs.

This file is only for quick comparison figures. It does not train models or
change any results. It reads summary JSON files that were created after feature
generation and turns them into a simple bar chart for inspection or slides.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_KEYS = [
    ("mean_token_nll", "Token NLL"),
    ("mean_self_consistency_disagreement", "Self-Consistency\nDisagreement"),
    ("mean_semantic_entropy", "Semantic\nEntropy"),
    ("mean_groundedness_score", "Groundedness"),
]


def _load_summary(path: Path) -> Dict:
    """Read one summary JSON file produced by the merge step."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _judge_unsupported_rate(summary: Dict) -> float:
    """Compute the unsupported fraction from the stored judge histogram."""
    histogram = summary.get("judge_label_histogram", {})
    supported = int(histogram.get("0", 0))
    unsupported = int(histogram.get("1", 0))
    total = supported + unsupported
    return (unsupported / total) if total else 0.0


def _dataset_label(path: Path, summary: Dict) -> str:
    """Infer a readable dataset label from the filename or JSON payload."""
    if "phantom" in path.name.lower():
        return "PHANTOM"
    if "wikiqa" in path.name.lower():
        return "WikiQA"
    return summary.get("dataset_name", path.stem)


def plot_comparison(summary_paths: List[Path], output_path: Path) -> None:
    """Plot feature means and unsupported rate for a small two dataset comparison."""
    summaries = [_load_summary(path) for path in summary_paths]
    labels = [_dataset_label(path, summary) for path, summary in zip(summary_paths, summaries)]

    metric_labels = [label for _, label in SUMMARY_KEYS]
    values = np.array(
        [[float(summary.get(key, 0.0) or 0.0) for key, _ in SUMMARY_KEYS] for summary in summaries],
        dtype=float,
    )
    unsupported_rates = np.array([_judge_unsupported_rate(summary) for summary in summaries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(metric_labels))
    width = 0.35 if len(labels) == 2 else max(0.2, 0.8 / max(len(labels), 1))

    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        axes[0].bar(x + offset, values[idx], width=width, label=label)

    axes[0].set_title("Feature Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel("Mean Value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    x2 = np.arange(len(labels))
    axes[1].bar(x2, unsupported_rates, width=0.5, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[1].set_title("Judge Unsupported Rate")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Unsupported Fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)

    for idx, rate in enumerate(unsupported_rates):
        axes[1].text(idx, rate + 0.03, f"{rate:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Smoke-Test Results Across Evidence Regimes", fontsize=14)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Keep CLI setup in one place so the script is easy to run from terminal."""
    parser = argparse.ArgumentParser(description="Plot comparison figure from two summary JSON files.")
    parser.add_argument("--phantom-summary", required=True)
    parser.add_argument("--wikiqa-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    plot_comparison(
        summary_paths=[Path(args.phantom_summary), Path(args.wikiqa_summary)],
        output_path=Path(args.output),
    )
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
