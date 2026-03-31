from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from features.evidence_consistency import evidence_consistency
from features.semantic_entropy import semantic_entropy
from features.self_consistency import self_consistency
from features.token_entropy import token_entropy


PROJECT_ROOT = CODE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
PHANTOM_OUTPUT_DIR = DATA_DIR / "phantom"
WIKIQA_OUTPUT_DIR = DATA_DIR / "wiki_qa"

DEFAULT_PHANTOM_LOCAL_FILE = PHANTOM_OUTPUT_DIR / "Phantom_10k_5000tokens_combined_first_100.jsonl"
DEFAULT_WIKIQA_LOCAL_FILE = WIKIQA_OUTPUT_DIR / "train_retrieved_first_100.jsonl"
DEFAULT_SMALL_MODEL = "qwen3:8b"
DEFAULT_JUDGE_MODEL = "qwen3:14b"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_SELF_CONSISTENCY_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_NUM_ROWS = 100
DEFAULT_K = 5


def _ensure_data_dirs() -> None:
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WIKIQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    num_predict: int,
    top_p: float,
    top_k: int,
    seed: Optional[int],
    ollama_url: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": 1,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": num_predict,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean_generated_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*(final answer|answer)\s*:\s*", "", cleaned).strip()
    return cleaned


def build_generation_messages(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are answering a grounded QA task. Use only the provided context. "
                "Return only the final answer text. If the answer is not supported by the context, "
                "reply exactly: The answer is not supported by the context."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nReturn one short final answer only.",
        },
    ]


def build_judge_messages(question: str, context: str, answer: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded hallucination judge. Decide whether the candidate answer "
                "is fully supported by the provided context only. Return strict JSON with keys "
                "label, binary_label, explanation. Use supported/0 if fully supported. "
                "Use unsupported/1 if contradicted, unverifiable, or adds unsupported facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion:\n{question}\n\nCandidate answer:\n{answer}\n\nReturn only JSON."
            ),
        },
    ]


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse judge output: {text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge output was not an object: {text}")
    return parsed


def _normalize_binary_judge_label(parsed: Dict[str, Any]) -> int:
    value = parsed.get("binary_label")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())

    label = str(parsed.get("label", "")).strip().lower()
    if label == "supported":
        return 0
    if label == "unsupported":
        return 1
    raise ValueError(f"Could not normalize judge output: {parsed}")


def _normalize_binary_dataset_label(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label)
    normalized = str(label).strip().lower()
    if normalized in {"0", "supported", "not hallucination", "non hallucination", "non-hallucination"}:
        return 0
    if normalized in {"1", "unsupported", "hallucination"}:
        return 1
    return None


def _load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if resolved_device == "cuda" else None,
        trust_remote_code=True,
    )
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, resolved_device


def _load_nli_resources(
    model_name: str = DEFAULT_NLI_MODEL,
    device: Optional[str] = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if resolved_device == "cuda" else None,
    )
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def _load_self_consistency_model(device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_SELF_CONSISTENCY_MODEL, device=_resolve_device(device))


def _parse_ollama_logprobs(response: Dict[str, Any]) -> Tuple[List[float], List[Dict[str, Any]]]:
    raw_logprobs = response.get("logprobs") or []
    token_logprobs: List[float] = []
    token_details: List[Dict[str, Any]] = []

    for entry in raw_logprobs:
        if isinstance(entry, dict):
            logprob = entry.get("logprob")
            if logprob is not None:
                token_logprobs.append(float(logprob))
            detail = {
                "token": entry.get("token"),
                "logprob": float(logprob) if logprob is not None else None,
                "entropy": entry.get("entropy"),
                "top_logprobs": entry.get("top_logprobs"),
            }
            token_details.append(detail)

    return token_logprobs, token_details


def generate_k_answers_via_api(
    model_name: str,
    question: str,
    context: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed_start: int,
    ollama_url: str,
) -> Dict[str, Any]:
    messages = build_generation_messages(question=question, context=context)

    answers: List[str] = []
    total_log_probs: List[Optional[float]] = []
    average_log_probs: List[Optional[float]] = []
    per_answer_token_details: List[List[Dict[str, Any]]] = []
    raw_generation_responses: List[Dict[str, Any]] = []

    for sample_idx in range(k):
        response = _ollama_chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            num_predict=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            seed=seed_start + sample_idx,
            ollama_url=ollama_url,
        )
        cleaned = _clean_generated_answer(response.get("message", {}).get("content", "") or "")
        token_logprobs, token_details = _parse_ollama_logprobs(response)

        answers.append(cleaned)
        total_log_probs.append(sum(token_logprobs) if token_logprobs else None)
        average_log_probs.append((sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None)
        per_answer_token_details.append(token_details)
        raw_generation_responses.append(response)

    return {
        "model_name": model_name,
        "answers": answers,
        "served_answer_index": 0,
        "served_answer": answers[0] if answers else "",
        "total_log_probs": total_log_probs,
        "average_log_probs": average_log_probs,
        "per_answer_token_details": per_answer_token_details,
        "raw_generation_responses": raw_generation_responses,
    }


def generate_k_answers_locally(
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed_start: int,
) -> Dict[str, Any]:
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    prompt_length = inputs["input_ids"].shape[1]

    answers: List[str] = []
    total_log_probs: List[Optional[float]] = []
    average_log_probs: List[Optional[float]] = []
    per_answer_token_details: List[List[Dict[str, Any]]] = []

    for sample_idx in range(k):
        torch.manual_seed(seed_start + sample_idx)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        sequence = outputs.sequences[0]
        generated_tokens = sequence[prompt_length:]
        answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )[0][: generated_tokens.shape[0]]

        token_details: List[Dict[str, Any]] = []
        for token_id, logprob in zip(generated_tokens.tolist(), transition_scores.tolist()):
            token_details.append(
                {
                    "token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "logprob": float(logprob),
                    "entropy": None,
                    "top_logprobs": None,
                }
            )

        answers.append(answer_text)
        total_log_prob = float(transition_scores.sum().item()) if generated_tokens.numel() else 0.0
        average_log_prob = total_log_prob / max(int(generated_tokens.numel()), 1)
        total_log_probs.append(total_log_prob)
        average_log_probs.append(average_log_prob)
        per_answer_token_details.append(token_details)

    return {
        "model_name": model_name,
        "answers": answers,
        "served_answer_index": 0,
        "served_answer": answers[0] if answers else "",
        "total_log_probs": total_log_probs,
        "average_log_probs": average_log_probs,
        "per_answer_token_details": per_answer_token_details,
        "raw_generation_responses": [],
    }


def generate_gt(
    big_model: str,
    our_answer: str,
    context: str,
    question: str,
    ollama_url: str,
    max_tokens: int = 200,
) -> Dict[str, Any]:
    response = _ollama_chat(
        model=big_model,
        messages=build_judge_messages(question=question, context=context, answer=our_answer),
        temperature=0.0,
        num_predict=max_tokens,
        top_p=1.0,
        top_k=50,
        seed=None,
        ollama_url=ollama_url,
    )
    judge_text = response.get("message", {}).get("content", "")
    parsed = _extract_json_object(judge_text)
    binary_label = _normalize_binary_judge_label(parsed)
    return {
        "judge_model": big_model,
        "raw_response": judge_text,
        "parsed_response": parsed,
        "binary_label": binary_label,
        "label": "unsupported" if binary_label == 1 else "supported",
    }


def load_jsonl_dataset(path: Path, dataset_name: str, num_rows: int) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if len(rows) >= num_rows:
                break
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if dataset_name == "phantom":
                rows.append(
                    {
                        "example_id": str(raw.get("id", raw.get("Unnamed: 0", idx))),
                        "dataset_name": "phantom",
                        "question": raw.get("query", ""),
                        "context": raw.get("context", ""),
                        "ground_truth_answer": raw.get("answer"),
                        "ground_truth_label": raw.get("ground_truth_label"),
                    }
                )
            else:
                question_id = raw.get("question_id", f"wikiqa_{idx}")
                rows.append(
                    {
                        "example_id": str(raw.get("id", f"{question_id}_{idx}")),
                        "dataset_name": "wikiqa",
                        "question": raw.get("query", raw.get("question", "")),
                        "context": raw.get("context", ""),
                        "ground_truth_answer": raw.get("answer"),
                        "ground_truth_label": raw.get("label", raw.get("ground_truth_label")),
                    }
                )
    return Dataset.from_list(rows)


def _default_data_file_for_dataset(dataset_name: str) -> Path:
    if dataset_name == "phantom":
        return DEFAULT_PHANTOM_LOCAL_FILE
    if dataset_name == "wikiqa":
        return DEFAULT_WIKIQA_LOCAL_FILE
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _default_output_file_for_dataset(dataset_name: str, model_name: str, num_rows: int, k: int) -> Path:
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
    if dataset_name == "phantom":
        return PHANTOM_OUTPUT_DIR / f"{sanitized_model_name}_k{k}_phantom_{num_rows}_features.json"
    return WIKIQA_OUTPUT_DIR / f"{sanitized_model_name}_k{k}_wikiqa_{num_rows}_features.json"


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


def summarize_results(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"num_examples": 0}

    summary: Dict[str, Any] = {"num_examples": len(records)}

    def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    summary["mean_token_nll"] = _mean(
        [row["token_uncertainty"].get("mean_token_nll") for row in records if row.get("token_uncertainty")]
    )
    summary["mean_self_consistency_disagreement"] = _mean(
        [row["self_consistency"].get("disagreement_score") for row in records if row.get("self_consistency")]
    )
    summary["mean_semantic_entropy"] = _mean(
        [row["semantic_entropy"].get("semantic_entropy") for row in records if row.get("semantic_entropy")]
    )
    summary["mean_groundedness_score"] = _mean(
        [row["evidence_consistency"].get("groundedness_score") for row in records if row.get("evidence_consistency")]
    )

    label_histogram: Dict[str, int] = {}
    for row in records:
        label = row.get("ground_truth_label")
        if label is not None:
            key = str(label)
            label_histogram[key] = label_histogram.get(key, 0) + 1
    if label_histogram:
        summary["ground_truth_label_histogram"] = label_histogram

    judge_histogram: Dict[str, int] = {}
    comparable_count = 0
    exact_match_count = 0
    for row in records:
        judge = row.get("judge_label")
        if not judge:
            continue
        judge_key = str(judge["binary_label"])
        judge_histogram[judge_key] = judge_histogram.get(judge_key, 0) + 1
        dataset_binary = _normalize_binary_dataset_label(row.get("ground_truth_label"))
        if dataset_binary is not None:
            comparable_count += 1
            if dataset_binary == int(judge["binary_label"]):
                exact_match_count += 1
    if judge_histogram:
        summary["judge_label_histogram"] = judge_histogram
    if comparable_count:
        summary["judge_vs_dataset_label_accuracy"] = exact_match_count / comparable_count

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PHANTOM or WikiQA answers and compute all four features.")
    parser.add_argument("--dataset", choices=["phantom", "wikiqa"], required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--model", default=DEFAULT_SMALL_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--local-generation", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    _ensure_data_dirs()

    output_file = Path(args.output_file) if args.output_file else _default_output_file_for_dataset(
        args.dataset,
        args.model,
        args.num_rows,
        args.k,
    )
    summary_file = output_file.with_name(f"{output_file.stem}_summary.json")

    data_file = Path(args.data_file) if args.data_file else _default_data_file_for_dataset(args.dataset)
    dataset = load_jsonl_dataset(data_file, dataset_name=args.dataset, num_rows=args.num_rows)

    evidence_model, evidence_tokenizer, _ = _load_nli_resources(device=args.device)
    self_consistency_model = _load_self_consistency_model(device=args.device)

    local_generation_model = None
    local_generation_tokenizer = None
    local_generation_device = None
    if args.local_generation:
        local_generation_model, local_generation_tokenizer, local_generation_device = _load_model_and_tokenizer(
            args.model,
            device=args.device,
        )

    records: List[Dict[str, Any]] = []
    for row in dataset:
        if args.local_generation:
            generation = generate_k_answers_locally(
                model_name=args.model,
                model=local_generation_model,
                tokenizer=local_generation_tokenizer,
                question=row["question"],
                context=row["context"],
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed_start=args.seed_start,
            )
        else:
            generation = generate_k_answers_via_api(
                model_name=args.model,
                question=row["question"],
                context=row["context"],
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed_start=args.seed_start,
                ollama_url=args.ollama_url,
            )

        semantic_log_probs = generation["total_log_probs"]
        if any(value is None for value in semantic_log_probs):
            semantic_log_probs = None

        served_answer_index = generation["served_answer_index"]
        served_answer = generation["served_answer"]
        served_token_details = generation["per_answer_token_details"][served_answer_index]
        served_token_logprobs = [
            detail["logprob"] for detail in served_token_details if detail.get("logprob") is not None
        ]

        token_uncertainty_result = token_entropy(
            token_logprobs=served_token_logprobs,
            token_details=served_token_details,
            tokenizer=local_generation_tokenizer if args.local_generation else None,
            model=local_generation_model if args.local_generation else None,
            answer=served_answer,
        )

        semantic_entropy_result = semantic_entropy(
            question=row["question"],
            k_sampled_answers=generation["answers"],
            log_probs_for_token_probabilities=semantic_log_probs,
        )

        evidence_consistency_result = evidence_consistency(
            tokenizer=evidence_tokenizer,
            model=evidence_model,
            context=row["context"],
            answer=served_answer,
        )

        self_consistency_result = self_consistency(
            sampled_answers=generation["answers"],
            sbert_model=self_consistency_model,
        )

        record: Dict[str, Any] = {
            "example_id": row["example_id"],
            "dataset_name": row["dataset_name"],
            "question": row["question"],
            "context": row["context"],
            "ground_truth_answer": row.get("ground_truth_answer"),
            "ground_truth_label": row.get("ground_truth_label"),
            "model_name": generation["model_name"],
            "answers": generation["answers"],
            "served_answer_index": served_answer_index,
            "served_answer": served_answer,
            "total_log_probs": generation["total_log_probs"],
            "average_log_probs": generation["average_log_probs"],
            "per_answer_token_details": generation["per_answer_token_details"],
            "raw_generation_responses": generation["raw_generation_responses"],
            "decoding_config": {
                "dataset": args.dataset,
                "model_name": args.model,
                "judge_model": None if args.no_judge else args.judge_model,
                "k": args.k,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "seed_start": args.seed_start,
                "ollama_url": args.ollama_url,
                "local_generation": args.local_generation,
            },
            "token_uncertainty": token_uncertainty_result,
            "self_consistency": self_consistency_result,
            "semantic_entropy": semantic_entropy_result,
            "evidence_consistency": evidence_consistency_result,
        }

        if not args.no_judge:
            record["judge_label"] = generate_gt(
                big_model=args.judge_model,
                our_answer=served_answer,
                context=row["context"],
                question=row["question"],
                ollama_url=args.ollama_url,
            )

        records.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    summary = summarize_results(records)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} {args.dataset} records to: {output_file}")
    print(f"Saved summary to: {summary_file}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
