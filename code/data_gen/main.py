from __future__ import annotations

"""Download and prepare local PHANTOM or WikiQA subsets.

This script is the data intake step. It fetches raw examples, optionally
retrieves Wikipedia context for WikiQA, and writes simple JSONL files that the
generation pipeline can read later.
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHANTOM_OUTPUT_DIR = DATA_DIR / "phantom"
WIKIQA_OUTPUT_DIR = DATA_DIR / "wiki_qa"
HF_CACHE_DIR = Path("C:/hf_rahd_cache")

DEFAULT_PHANTOM_REPO = "seyled/Phantom_Hallucination_Detection"
DEFAULT_PHANTOM_FILES = [
    "PhantomDataset/Phantom_10k_2000tokens_beginning.csv",
    "PhantomDataset/Phantom_10k_2000tokens_middle.csv",
    "PhantomDataset/Phantom_10k_2000tokens_end.csv",
    "PhantomDataset/Phantom_10k_5000tokens_beginning.csv",
    "PhantomDataset/Phantom_10k_5000tokens_middle.csv",
    "PhantomDataset/Phantom_10k_5000tokens_end.csv",
    "PhantomDataset/Phantom_10k_10000tokens_beginning.csv",
    "PhantomDataset/Phantom_10k_10000tokens_middle.csv",
    "PhantomDataset/Phantom_10k_10000tokens_end.csv",
    "PhantomDataset/Phantom_10k_20000tokens_beginning.csv",
    "PhantomDataset/Phantom_10k_20000tokens_middle.csv",
    "PhantomDataset/Phantom_10k_20000tokens_end.csv",
    "PhantomDataset/Phantom_10k_30000tokens_beginning.csv",
    "PhantomDataset/Phantom_10k_30000tokens_middle.csv",
    "PhantomDataset/Phantom_10k_30000tokens_end.csv",
]
DEFAULT_WIKIQA_REPO = "microsoft/wiki_qa"
DEFAULT_NUM_ROWS = 100


def _ensure_data_dirs() -> None:
    """Create the expected local data directories if they do not exist yet."""
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WIKIQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_local_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file when present."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    """Write iterable rows to JSONL so later stages can stream them easily."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_phantom_dataset(
    dataset_repo: str = DEFAULT_PHANTOM_REPO,
    data_files: List[str] | None = None,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> Dataset:
    """Load a de-duplicated PHANTOM subset from the Hugging Face dataset repo."""
    _ensure_data_dirs()
    if data_files is None:
        data_files = DEFAULT_PHANTOM_FILES

    selected_rows: List[dict] = []
    seen_prompt_label_keys: Set[Tuple[str, str, str]] = set()
    total_examples = 0

    for data_file in data_files:
        csv_path = hf_hub_download(
            repo_id=dataset_repo,
            filename=data_file,
            repo_type="dataset",
            cache_dir=str(HF_CACHE_DIR),
            token=os.environ.get("HF_TOKEN"),
        )
        dataset_dict = load_dataset(
            "csv",
            data_files=csv_path,
            cache_dir=str(HF_CACHE_DIR),
        )
        dataset = dataset_dict["train"]
        available = len(dataset)
        taken_from_file = 0

        for row in dataset:
            prompt_label_key = (
                str(row.get("query") or "").strip(),
                str(row.get("context") or "").strip(),
                str(row.get("ground_truth_label") or "").strip(),
            )
            if prompt_label_key in seen_prompt_label_keys:
                continue
            seen_prompt_label_keys.add(prompt_label_key)
            selected_rows.append(
                {
                    "Unnamed: 0": row.get("Unnamed: 0"),
                    "query": row.get("query"),
                    "context": row.get("context"),
                    "answer": row.get("answer"),
                    "ground_truth_label": row.get("ground_truth_label"),
                }
            )
            total_examples += 1
            taken_from_file += 1
            if total_examples >= num_rows:
                break

        print(
            f"[PHANTOM] {Path(data_file).name}: available={available}, "
            f"added_unique_prompt_labels={taken_from_file}, total_unique_prompt_labels={total_examples}"
        )

        if total_examples >= num_rows:
            break

    if not selected_rows:
        raise ValueError("No PHANTOM examples were loaded.")

    return Dataset.from_list(selected_rows[:num_rows])


def download_wikiqa_dataset(
    dataset_repo: str = DEFAULT_WIKIQA_REPO,
    split: str = "train",
) -> Dataset:
    """Download a WikiQA split through the datasets library."""
    _ensure_data_dirs()
    dataset = load_dataset(
        dataset_repo,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )
    return dataset


def make_api_call_to_wikimedia(title: str) -> str:
    """Fetch plain-text Wikipedia content for one page title.

    This is used to attach evidence context to WikiQA questions so groundedness
    features can be computed later.
    """
    if not str(title or "").strip():
        return ""

    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    query = urllib.parse.urlencode(params)
    url = f"https://en.wikipedia.org/w/api.php?{query}"
    headers = {"User-Agent": "Risk-Adjusted-Hallucinations/1.0"}

    for attempt in range(5):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            return page.get("extract", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return ""
        except Exception:
            return ""
    return ""


def process_wikiqa_with_retrieval(
    split: str = "train",
    num_rows: int = DEFAULT_NUM_ROWS,
) -> List[dict]:
    """Attach retrieved Wikipedia context to WikiQA rows and keep unique prompts."""
    wikiqa = download_wikiqa_dataset(split=split)
    retrieved_rows: List[dict] = []
    seen_prompt_keys: Set[Tuple[str, str]] = set()
    context_cache: dict[str, str] = {}

    for idx, row in enumerate(wikiqa):
        if idx > 0 and idx % 500 == 0:
            print(
                f"[WikiQA] scanned={idx} kept={len(retrieved_rows)} "
                f"cached_titles={len(context_cache)}"
            )

        normalized_row = dict(row)
        normalized_row["question_id"] = row.get("question_id", f"{split}_{idx}")
        document_title = str(row.get("document_title") or "")
        if document_title not in context_cache:
            context_cache[document_title] = make_api_call_to_wikimedia(document_title)
        normalized_row["context"] = context_cache[document_title]
        if not str(normalized_row["context"] or "").strip():
            continue
        prompt_key = (
            str(normalized_row.get("question") or "").strip(),
            str(normalized_row.get("document_title") or "").strip(),
        )
        if prompt_key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(prompt_key)
        retrieved_rows.append(normalized_row)
        if len(retrieved_rows) % 100 == 0:
            print(
                f"[WikiQA] kept={len(retrieved_rows)}/{num_rows} "
                f"latest_title={document_title}"
            )
        if len(retrieved_rows) >= num_rows:
            break

    return retrieved_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for the dataset preparation step."""
    parser = argparse.ArgumentParser(description="Prepare PHANTOM or WikiQA local jsonl subsets.")
    parser.add_argument("--dataset", choices=["phantom", "wikiqa"], required=True)
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--split", default="train", help="WikiQA split to download.")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve Wikipedia evidence for WikiQA.")
    return parser


def main() -> None:
    """CLI entry point for preparing source data files."""
    args = _build_arg_parser().parse_args()
    _load_local_env()
    _ensure_data_dirs()

    if args.dataset == "phantom":
        dataset = load_phantom_dataset(num_rows=args.num_rows)
        output_path = PHANTOM_OUTPUT_DIR / f"Phantom_10k_5000tokens_combined_first_{len(dataset)}.jsonl"
        save_jsonl((dict(row) for row in dataset), output_path)
        print(f"Saved PHANTOM subset to: {output_path}")
        return

    if args.dataset == "wikiqa":
        if args.retrieve:
            rows = process_wikiqa_with_retrieval(split=args.split, num_rows=args.num_rows)
            output_path = WIKIQA_OUTPUT_DIR / f"{args.split}_retrieved_first_{len(rows)}.jsonl"
            save_jsonl(rows, output_path)
            print(f"Saved WikiQA retrieved subset to: {output_path}")
        else:
            dataset = download_wikiqa_dataset(split=args.split)
            limited = dataset.select(range(min(args.num_rows, len(dataset))))
            output_path = WIKIQA_OUTPUT_DIR / f"{args.split}_first_{len(limited)}.jsonl"
            save_jsonl((dict(row) for row in limited), output_path)
            print(f"Saved WikiQA raw subset to: {output_path}")


if __name__ == "__main__":
    main()
