from __future__ import annotations

"""Split a large JSONL file into smaller JSONL chunks.

This is a helper for long generation runs. It does not change content, only how
the file is partitioned on disk.
"""

import argparse
import math
from pathlib import Path
from typing import List


def _read_jsonl_lines(path: Path) -> List[str]:
    """Read non-empty JSONL lines exactly as stored."""
    with path.open("r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


def split_jsonl(input_path: Path, output_dir: Path, num_splits: int, prefix: str) -> None:
    """Write evenly sized JSONL chunks for easier batched processing."""
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")

    lines = _read_jsonl_lines(input_path)
    if not lines:
        raise ValueError(f"No non-empty JSONL rows found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(lines)
    chunk_size = math.ceil(total_rows / num_splits)

    print(f"Input rows: {total_rows}")
    print(f"Requested splits: {num_splits}")
    print(f"Chunk size: {chunk_size}")

    for split_idx in range(num_splits):
        start = split_idx * chunk_size
        end = min(start + chunk_size, total_rows)
        if start >= total_rows:
            break

        chunk_lines = lines[start:end]
        output_path = output_dir / f"{prefix}_part_{split_idx + 1}_of_{num_splits}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.writelines(chunk_lines)

        print(
            f"Wrote split {split_idx + 1}/{num_splits}: "
            f"rows={len(chunk_lines)} start={start} end={end - 1} -> {output_path}"
        )


def main() -> None:
    """CLI entry point for splitting a JSONL file into smaller pieces."""
    parser = argparse.ArgumentParser(description="Split a JSONL file into multiple smaller JSONL files.")
    parser.add_argument("--input", required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where split files will be written.")
    parser.add_argument("--num-splits", type=int, default=5, help="Number of output chunks to create.")
    parser.add_argument(
        "--prefix",
        default="split",
        help="Filename prefix for generated chunks. Example: phantom_5000",
    )
    args = parser.parse_args()

    split_jsonl(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        num_splits=args.num_splits,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
