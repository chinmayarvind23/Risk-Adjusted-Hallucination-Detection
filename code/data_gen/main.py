from datasets import Dataset, load_dataset
from pathlib import Path
import json
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHANTOM_OUTPUT_DIR = DATA_DIR / "phantom"
WIKIQA_OUTPUT_DIR = DATA_DIR / "wiki_qa"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

DEFAULT_PHANTOM_REPO = "seyled/Phantom_Hallucination_Detection"
DEFAULT_PHANTOM_FILE = "PhantomDataset/Phantom_10k_seed.csv"
DEFAULT_WIKIQA_REPO = "microsoft/wiki_qa"
DEFAULT_NUM_ROWS = 100


def _ensure_data_dirs() -> None:
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WIKIQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_phantom_dataset(
    dataset_repo: str = DEFAULT_PHANTOM_REPO,
    data_file: str = DEFAULT_PHANTOM_FILE,
    num_rows: int = DEFAULT_NUM_ROWS,
    save_subset: bool = True,
) -> Dataset:
    _ensure_data_dirs()

    dataset_dict = load_dataset(
        dataset_repo,
        data_files=data_file,
        cache_dir=str(HF_CACHE_DIR),
    )

    if "train" not in dataset_dict:
        raise ValueError(f"Expected 'train' split, found: {list(dataset_dict.keys())}")

    dataset = dataset_dict["train"]
    limited_dataset = dataset.select(range(min(num_rows, len(dataset))))

    if save_subset:
        subset_name = Path(data_file).stem
        subset_path = PHANTOM_OUTPUT_DIR / f"{subset_name}_first_{len(limited_dataset)}.jsonl"
        save_jsonl([dict(row) for row in limited_dataset], subset_path)

    return limited_dataset


def download_wikiqa_dataset(
    dataset_repo: str = DEFAULT_WIKIQA_REPO,
    split: str = "train",
    num_rows: int = DEFAULT_NUM_ROWS,
    save_subset: bool = True,
) -> Dataset:
    _ensure_data_dirs()

    dataset = load_dataset(
        dataset_repo,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )

    limited_dataset = dataset.select(range(min(num_rows, len(dataset))))

    if save_subset:
        subset_path = WIKIQA_OUTPUT_DIR / f"{split}_first_{len(limited_dataset)}.jsonl"
        save_jsonl([dict(row) for row in limited_dataset], subset_path)

    return limited_dataset


# def make_api_call_to_wikimedia(question: str, title: str) -> dict:
#     headers = {"User-Agent": "Risk-Adjusted-Hallucinations"}
#     params = {
#         "action": "query",
#         "titles": title,
#         "prop": "extracts",
#         "explaintext": True,
#         "format": "json",
#     }

#     try:
#         response = requests.get(
#             "https://en.wikipedia.org/w/api.php",
#             params=params,
#             headers=headers,
#             timeout=20,
#         )
#         response.raise_for_status()
#         data = response.json()
#         pages = data["query"]["pages"]
#         page = next(iter(pages.values()))
#         context = page.get("extract", "")
#     except Exception:
#         context = ""

#     return {
#         "question": question,
#         "document_title": title,
#         "context": context,
#     }

def make_api_call_to_wikimedia(title: str) -> str:
    headers = {"User-Agent": "Risk-Adjusted-Hallucinations"}
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }

    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "")
    except Exception:
        return ""


# def retrieve_evidence_wqa(qt_jsons):
#     result = []
#     for qt in qt_jsons:
#         question = qt["question"]
#         title = qt["document_title"]
#         evidence = make_api_call_to_wikimedia(question, title)
#         result.append(evidence)
#     return result
def retrieve_evidence_wqa(rows):
    result = []
    for row in rows:
        new_row = dict(row)
        new_row["context"] = make_api_call_to_wikimedia(row["document_title"])
        result.append(new_row)
    return result


def prepare_wikiqa_for_retrieval(wikiqa_dataset: Dataset):
    rows = [dict(row) for row in wikiqa_dataset]

    if not rows:
        return []

    print("WikiQA columns:", list(rows[0].keys()))

    retrieval_inputs = []
    skipped = 0

    for row in rows:
        question = row.get("question", "")
        title = row.get("document_title", "")

        if not question or not title:
            skipped += 1
            continue

        retrieval_inputs.append(dict(row))   # keep full row

    print(f"Prepared {len(retrieval_inputs)} rows for retrieval, skipped {skipped}")
    return retrieval_inputs


def process_wikiqa_with_retrieval(
    split: str = "train",
    num_rows: int = DEFAULT_NUM_ROWS,
    save_raw_subset: bool = True,
    save_retrieved_subset: bool = True,
):
    wikiqa = download_wikiqa_dataset(
        split=split,
        num_rows=num_rows,
        save_subset=save_raw_subset,
    )

    retrieval_inputs = prepare_wikiqa_for_retrieval(wikiqa)
    retrieved_rows = retrieve_evidence_wqa(retrieval_inputs)

    if save_retrieved_subset:
        out_path = WIKIQA_OUTPUT_DIR / f"{split}_retrieved_first_{len(retrieved_rows)}.jsonl"
        save_jsonl(retrieved_rows, out_path)

    return wikiqa, retrieved_rows


if __name__ == "__main__":
    phantom = load_phantom_dataset()
    wikiqa_raw, wikiqa_retrieved = process_wikiqa_with_retrieval()

    print("PHANTOM rows:", len(phantom))
    print("WikiQA raw rows:", len(wikiqa_raw))
    print("WikiQA retrieved rows:", len(wikiqa_retrieved))

    if len(phantom) > 0:
        print("\nPHANTOM sample keys:", list(dict(phantom[0]).keys()))
        print("PHANTOM sample row:", dict(phantom[0]))

    if len(wikiqa_raw) > 0:
        print("\nWikiQA raw sample keys:", list(dict(wikiqa_raw[0]).keys()))
        print("WikiQA raw sample row:", dict(wikiqa_raw[0]))

    if len(wikiqa_retrieved) > 0:
        print("\nWikiQA retrieved sample keys:", list(wikiqa_retrieved[0].keys()))
        print("WikiQA retrieved sample row:", wikiqa_retrieved[0])