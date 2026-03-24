from datasets import Dataset, load_dataset
import os
from pathlib import Path
import json
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHANTOM_OUTPUT_DIR = DATA_DIR / "phantom"
HF_CACHE_DIR = Path("C:/hf_rahd_cache")

DEFAULT_PHANTOM_REPO = "seyled/Phantom_Hallucination_Detection"
DEFAULT_PHANTOM_FILE = "PhantomDataset/Phantom_10k_seed.csv"
DEFAULT_SMALL_MODEL = "qwen3:8b"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_JUDGE_MODEL = "qwen3:14b"
DEFAULT_NUM_ROWS = 100
DEFAULT_K = 3
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

WIKIQA_OUTPUT_DIR = DATA_DIR / "wiki_qa"
DEFAULT_WIKIQA_REPO = "microsoft/wiki_qa"


def _ensure_data_dirs() -> None:
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    dataset = dataset_dict["train"]
    limited_dataset = dataset.select(range(min(num_rows, len(dataset))))
    if save_subset:
        subset_name = Path(data_file).stem
        subset_path = PHANTOM_OUTPUT_DIR / f"{subset_name}_first_{len(limited_dataset)}.jsonl"
        with subset_path.open("w", encoding="utf-8") as handle:
            for row in limited_dataset:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return limited_dataset

def download_wikiqa_dataset(
    dataset_repo: str = DEFAULT_WIKIQA_REPO,
    split: str = "train",
    num_rows: int = DEFAULT_NUM_ROWS,
    save_subset: bool = True,
    ) -> Dataset:
    """
    Downloads the wiki QA dataset and stores in the data directory
    """
    _ensure_data_dirs()
    WIKIQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_repo,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )

    limited_dataset = dataset.select(range(min(num_rows, len(dataset))))

    if save_subset:
        subset_path = WIKIQA_OUTPUT_DIR / f"{split}_first_{len(limited_dataset)}.jsonl"
        with subset_path.open("w", encoding="utf-8") as handle:
            for row in limited_dataset:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    return limited_dataset

def make_api_call_to_wikimedia(question, title):
    header = {"User-Agent": "Risk-Adjusted-Halluciations"}
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,                
        "format": "json"
    }
    result = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=header).json()["query"]["pages"]
    page = next(iter(result.values()))
    return {"question": question, "document_title": title, "context":page.get("extract", "")}

def retrieve_evidence_wqa(qt_jsons):
    result = []
    for qt in qt_jsons:
        question = qt["question"]
        title = qt["document_title"]
        evidence = make_api_call_to_wikimedia(question, title)
        result.append(evidence)
    
    return json.dumps(result, indent=4)