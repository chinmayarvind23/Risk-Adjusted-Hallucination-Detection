from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from features.semantic_entropy import semantic_entropy
from features.token_entropy import token_entropy
from features.evidence_consistency import evidence_consistency
from features.self_consistency import self_consistency

from sentence_transformers import SentenceTransformer

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHANTOM_OUTPUT_DIR = DATA_DIR / "phantom"

WIKIQA_OUTPUT_DIR = DATA_DIR / "wiki_qa"
DEFAULT_WIKIQA_LOCAL_FILE = WIKIQA_OUTPUT_DIR / "train_retrieved_first_100.jsonl"

HF_CACHE_DIR = Path("C:/hf_rahd_cache")

DEFAULT_PHANTOM_REPO = "seyled/Phantom_Hallucination_Detection"
DEFAULT_PHANTOM_LOCAL_FILE = PHANTOM_OUTPUT_DIR / "Phantom_10k_seed_first_100.jsonl"
DEFAULT_SMALL_MODEL = "qwen3:8b"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_JUDGE_MODEL = "qwen3:14b"
DEFAULT_NUM_ROWS = 100
DEFAULT_K = 3
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

def _ensure_data_dirs() -> None:
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WIKIQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)



#### TOKEN ENTROPY
DEFAULT_TOKEN_ENTROPY_MODEL = "Qwen/Qwen2.5-7B-Instruct"

"""
This is pretty much just for the Hugginface API key. Loads the .env
"""
def _load_local_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
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


### SELF CONSISTENCY
DEFAULT_SELF_CONSISTENCY_MODEL = "all-MiniLM-L6-v2"


"""
This is just to hit Ollama. Given a model and the prompt, we get the answer
Handles the OS-specific nuances insde the function
"""
def _ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    num_predict: int = 256,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:

    # Detect if the endpoint is /v1/completions (OpenAI-style) or /api/chat (Ollama)
    if "/v1/completions" in ollama_url:
        # OpenAI-style payload
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": num_predict,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": 1,
            "n": 1,
        }
        if seed is not None:
            payload["seed"] = seed
    else:
        # Ollama local API chat payload
        payload = {
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

    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return {"message": {"content": ""}, "logprobs": []}
    


"""
Setting device
"""
def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


"""
Given a model_id/name and tokenizer_id/name, loads it and returns them along with the device (uses resolve device)
"""
def _load_model_and_tokenizer(
    model_ours: Union[str, AutoModelForCausalLM],
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    resolved_device = _resolve_device(device)

    if isinstance(model_ours, str):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_ours, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ours,
            torch_dtype="auto",
            device_map="auto" if resolved_device == "cuda" else None,
            trust_remote_code=True,
        )
        if resolved_device != "cuda":
            model = model.to(resolved_device)
    else:
        model = model_ours
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when model_ours is already an instantiated model")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, resolved_device



def _load_token_entropy_resources(
    model_name: str = DEFAULT_TOKEN_ENTROPY_MODEL,
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


def _load_evidence_consistency_resources(
    model_name: str = DEFAULT_NLI_MODEL,
    device: Optional[str] = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
    resolved_device = _resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if resolved_device == "cuda" else None,
        trust_remote_code=True,
    )

    if resolved_device != "cuda":
        model = model.to(resolved_device)

    model.eval()
    return model, tokenizer, resolved_device

def _load_self_consistency_resources(
    model_name: str = DEFAULT_SELF_CONSISTENCY_MODEL,
    device: Optional[str] = None,
) -> SentenceTransformer:
    resolved_device = _resolve_device(device)
    return SentenceTransformer(model_name, device=resolved_device)


"""
Creating the prompt for the Phantom task
"""
def build_phantom_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the provided context. "
        "If the context does not support an answer, say that the answer is not supported.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


"""
Prompting for the LLM Judge to check whether the smaller model has given the right answer or not
"""
def build_judge_messages(question: str, context: str, answer: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded hallucination judge. "
                "Decide whether the candidate answer is supported by the provided context only. "
                "Return strict JSON with keys: label, binary_label, explanation. "
                "Use label='supported' and binary_label=0 if the answer is fully supported by the context. "
                "Use label='unsupported' and binary_label=1 if the answer is contradicted, adds unsupported facts, "
                "or cannot be verified from the context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                f"Candidate answer:\n{answer}\n\n"
                "Return only JSON."
            ),
        },
    ]


"""
Building the prompt for giving the {question, context} to the smaller models and generating an answer using that
"""
def build_generation_messages(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are answering a grounded question-answering task. "
                "Use only the provided context. "
                "Do not show your reasoning. Do not use <think> tags. "
                "Return only the final answer text. "
                "If the context does not support an answer, reply exactly: The answer is not supported by the context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Return one short final answer only. No reasoning. No bullet points. No preamble."
            ),
        },
    ]


"""
Cleaning up the answer generated by the smaller model
"""
def _clean_generated_answer(text: str) -> str:
    cleaned = text.strip()

    # Drop any explicit reasoning blocks
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()

    # Remove common answer wrappers/preambles
    cleaned = re.sub(r"(?im)^\s*(final answer|answer)\s*:\s*", "", cleaned).strip()

    # If reasoning leaked before final answer, keep only trailing answer-like segment
    split_markers = [
        "final answer:",
        "answer:",
    ]
    lowered = cleaned.lower()
    for marker in split_markers:
        idx = lowered.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].strip()
            lowered = cleaned.lower()

    return cleaned


"""
Just a utils function. No literal utility
"""
def _validate_nonempty_answers(answers: Sequence[str], model_name: str) -> None:
    if answers and all(not answer.strip() for answer in answers):
        raise ValueError(
            f"{model_name} returned empty final answers after cleaning. "
            "This usually means the model is still emitting thinking text instead of final content."
        )

"""
Parse a JSON object
"""
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
        raise ValueError(f"Could not parse judge output as JSON: {text}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge output JSON was not an object: {text}")
    return parsed


"""
Parsing the answer from the judge LLM
"""
def _normalize_binary_judge_label(parsed_judge_output: Dict[str, Any]) -> int:
    if "binary_label" in parsed_judge_output:
        binary_label = parsed_judge_output["binary_label"]
        if isinstance(binary_label, bool):
            return int(binary_label)
        if isinstance(binary_label, (int, float)):
            return int(binary_label)
        if isinstance(binary_label, str) and binary_label.strip() in {"0", "1"}:
            return int(binary_label.strip())

    label = str(parsed_judge_output.get("label", "")).strip().lower()
    if label == "supported":
        return 0
    if label == "unsupported":
        return 1

    raise ValueError(f"Could not normalize judge label from output: {parsed_judge_output}")


"""
Parsing the answer from the judge LLM for the exact yes/no
"""
def _normalize_binary_dataset_label(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label)

    normalized = str(label).strip().lower()
    if normalized in {"0", "supported", "not hallucination", "non-hallucination", "non hallucination"}:
        return 0
    if normalized in {"1", "unsupported", "hallucination"}:
        return 1

    return None


"""

initialize phantom json file

for phantom:
    for each question:
        [score1, score2, score3] = semantic_entropy()
        token_ent = token_entropy()
        token_ent = token_entropy()
        token_ent = token_entropy()


        write_to_json_file(f1, f2, f3, f4)


Finally, we have a full json file

load_json_into_tensors
load_tensors_
        
"""



"""
Generate the grount truth
"""
def generate_gt(
    big_model: str = DEFAULT_JUDGE_MODEL,
    our_answer: str = "",
    context: str = "",
    question: str = "",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_tokens: int = 200,
) -> Dict[str, Any]:
    """
    Use a larger API-hosted judge model to decide whether an answer is
    supported by the given context for a PHANTOM question
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if not context.strip():
        raise ValueError("context must be a non-empty string")
    if not our_answer.strip():
        raise ValueError("our_answer must be a non-empty string")

    completion = _ollama_chat(
        model=big_model,
        messages=build_judge_messages(question=question, context=context, answer=our_answer),
        num_predict=max_tokens,
        temperature=0.0,
        ollama_url=ollama_url,
    )
    judge_text = completion["message"]["content"]
    parsed_output = _extract_json_object(judge_text)
    binary_label = _normalize_binary_judge_label(parsed_output)

    return {
        "judge_model": big_model,
        "raw_response": judge_text,
        "parsed_response": parsed_output,
        "binary_label": binary_label,
        "label": "unsupported" if binary_label == 1 else "supported",
    }


"""
Loading the phantom dataset
"""

def load_phantom_dataset(
    local_file: Optional[Union[str, Path]] = None,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> Dataset:
    _ensure_data_dirs()

    if local_file is None:
        local_file = DEFAULT_PHANTOM_LOCAL_FILE
    else:
        local_file = Path(local_file)

    if not local_file.exists():
        raise FileNotFoundError(f"PHANTOM local file not found: {local_file}")

    rows = []
    with local_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rows = rows[:num_rows]
    return Dataset.from_list(rows)



def load_wikiqa_dataset(
    local_file: Optional[Union[str, Path]] = None,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> Dataset:
    _ensure_data_dirs()

    if local_file is None:
        local_file = DEFAULT_WIKIQA_LOCAL_FILE
    else:
        local_file = Path(local_file)

    if not local_file.exists():
        raise FileNotFoundError(f"WikiQA local file not found: {local_file}")

    rows = []
    with local_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            question_id = row.get("question_id", "unknown_qid")

            normalized_row = dict(row)
            normalized_row["id"] = f"{question_id}_{idx}"
            normalized_row["query"] = row.get("question", "")
            normalized_row["ground_truth_label"] = row.get("label")

            rows.append(normalized_row)

            if len(rows) >= num_rows:
                break

    return Dataset.from_list(rows)

def get_output_dir_for_dataset(dataset_name: str) -> Path:
    if dataset_name == "phantom":
        return PHANTOM_OUTPUT_DIR
    if dataset_name == "wikiqa":
        return WIKIQA_OUTPUT_DIR
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")



def load_dataset_for_pipeline(
    dataset_name: str,
    data_file: Optional[Union[str, Path]] = None,
    num_rows: int = DEFAULT_NUM_ROWS,
) -> Dataset:
    if dataset_name == "phantom":
        return load_phantom_dataset(local_file=data_file, num_rows=num_rows)
    if dataset_name == "wikiqa":
        return load_wikiqa_dataset(local_file=data_file, num_rows=num_rows)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")



"""
Function to get k answers from Ollama: This is used by the "generate_k_answers" function
That is the main function involved in the pipeline. This just abstracts out the ollama call
"""
def generate_k_answers_via_api(
    model_name: str = DEFAULT_SMALL_MODEL,
    question: str = "",
    context: str = "",
    k: int = DEFAULT_K,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    seed_start: int = 1234,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if not context.strip():
        raise ValueError("context must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be at least 1")

    prompt = build_phantom_prompt(question=question, context=context)
    messages = build_generation_messages(question=question, context=context)

    sampled_answers: List[str] = []
    average_log_probs: List[Optional[float]] = []
    total_log_probs: List[Optional[float]] = []
    raw_generation_responses: List[str] = []

    for sample_idx in range(k):
        response = _ollama_chat(
            model=model_name,
            messages=messages,
            num_predict=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed_start + sample_idx,
            top_k=top_k,
            ollama_url=ollama_url,
        )

        generated_text = _clean_generated_answer(
            response.get("message", {}).get("content", "") or ""
        )

        response_logprobs = response.get("logprobs") or []
        token_log_probs: List[float] = [
            float(tok.get("logprob")) for tok in response_logprobs if tok.get("logprob") is not None
        ]

        if token_log_probs:
            total_log_prob = float(sum(token_log_probs))
            avg_log_prob = total_log_prob / len(token_log_probs)
        else:
            total_log_prob = None
            avg_log_prob = None

        sampled_answers.append(generated_text)
        total_log_probs.append(total_log_prob)
        average_log_probs.append(avg_log_prob)
        raw_generation_responses.append(json.dumps(response, ensure_ascii=False))

    _validate_nonempty_answers(sampled_answers, model_name=model_name)

    result = {
        "model_name": model_name,
        "question": question,
        "context": context,
        "k": k,
        "answers": sampled_answers,
        "average_log_probs": average_log_probs,
        "total_log_probs": total_log_probs,
        "prompt": prompt,
        "generation_backend": "ollama",
        "raw_generation_responses": raw_generation_responses,
        "served_answer_index": 0,
        "served_answer": sampled_answers[0] if sampled_answers else "",
        "decoding_config": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "k": k,
            "seed_start": seed_start,
            "ollama_url": ollama_url,
        },
    }

    return result



"""
Generate k answers for each question
"""
def generate_k_answers(
    model_ours: Union[str, AutoModelForCausalLM] = DEFAULT_SMALL_MODEL,
    question: str = "",
    context: str = "",
    k: int = DEFAULT_K,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    use_api_generation: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    seed_start: int = 1234,  # added for reproducibility
) -> Dict[str, Any]:
    """
    Generate k sampled answers for one PHANTOM question-context pair and
    return the answers together with average token log-probabilities and decoding metadata.
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if not context.strip():
        raise ValueError("context must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be at least 1")

    if use_api_generation:
        if not isinstance(model_ours, str):
            raise ValueError("API generation requires model_ours to be a Hugging Face model id string")
        gen_result = generate_k_answers_via_api(
            model_name=model_ours,
            question=question,
            context=context,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ollama_url=ollama_url,
            seed_start=seed_start
        )
    else:
        model, tokenizer, resolved_device = _load_model_and_tokenizer(
            model_ours=model_ours,
            tokenizer=tokenizer,
            device=device,
        )

        prompt = build_phantom_prompt(question=question, context=context)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        torch.manual_seed(seed_start)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                num_return_sequences=k,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )

        sampled_answers: List[str] = []
        average_log_probs: List[float] = []
        total_log_probs: List[float] = []

        for sequence, token_scores in zip(outputs.sequences, transition_scores):
            generated_tokens = sequence[prompt_length:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            valid_scores = token_scores[: generated_tokens.shape[0]]
            total_log_prob = float(valid_scores.sum().item()) if generated_tokens.numel() else 0.0
            avg_log_prob = total_log_prob / max(int(generated_tokens.numel()), 1)

            sampled_answers.append(answer_text)
            total_log_probs.append(total_log_prob)
            average_log_probs.append(avg_log_prob)

        gen_result = {
            "model_name": model_ours if isinstance(model_ours, str) else getattr(model.config, "_name_or_path", DEFAULT_SMALL_MODEL),
            "question": question,
            "context": context,
            "k": k,
            "answers": sampled_answers,
            "average_log_probs": average_log_probs,
            "total_log_probs": total_log_probs,
            "prompt": prompt,
        }

    gen_result["served_answer_index"] = 0
    gen_result["served_answer"] = gen_result["answers"][0] if gen_result["answers"] else ""
    gen_result["decoding_config"] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "k": k,
        "seed_start": seed_start,
        "use_api_generation": use_api_generation,
        "ollama_url": ollama_url,
    }

    return gen_result



"""
Generating answers for the Phantom dataset
"""
def generate_answers_for_phantom_subset(
    model_ours: Union[str, AutoModelForCausalLM] = DEFAULT_SMALL_MODEL,
    data_file: str = DEFAULT_PHANTOM_LOCAL_FILE,
    num_rows: int = DEFAULT_NUM_ROWS,
    k: int = DEFAULT_K,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
    judge_model: Optional[str] = DEFAULT_JUDGE_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    use_api_generation: bool = True,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run sampled generation over the first num_rows PHANTOM examples.
    """
    dataset = load_phantom_dataset(local_file=data_file, num_rows=num_rows)

    if use_api_generation:
        model = None
        resolved_device = device
    else:
        model, tokenizer, resolved_device = _load_model_and_tokenizer(
            model_ours=model_ours,
            tokenizer=tokenizer,
            device=device,
        )

    generations: List[Dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        example_id = str(row.get("id", row.get("Unnamed: 0", idx)))
        generation = generate_k_answers(
            model_ours=model_ours if use_api_generation else model,
            tokenizer=tokenizer,
            device=resolved_device,
            question=row["query"],
            context=row["context"],
            k=k,
            use_api_generation=use_api_generation,
            ollama_url=ollama_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generation["example_id"] = example_id
        generation["ground_truth_answer"] = row.get("answer")
        generation["ground_truth_label"] = row.get("ground_truth_label")
        served_answer = generation["answers"][0] if generation["answers"] else ""
        generation["served_answer"] = served_answer
        semantic_log_probs = generation["total_log_probs"]
        if any(log_prob is None for log_prob in semantic_log_probs):
            semantic_log_probs = None

        generation["semantic_entropy"] = semantic_entropy(
            question=row["query"],
            k_sampled_answers=generation["answers"],
            log_probs_for_token_probabilities=semantic_log_probs,
        )
        if judge_model:
            generation["judge_label"] = generate_gt(
                big_model=judge_model,
                our_answer=served_answer,
                context=row["context"],
                question=row["query"],
                ollama_url=ollama_url,
            )
        generations.append(generation)

    sanitized_model_name = str(model_ours).replace("/", "_").replace(":", "_")
    output_path = PHANTOM_OUTPUT_DIR / f"{sanitized_model_name}_k{k}_{Path(data_file).stem}_first_{num_rows}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(generations, handle, indent=2, ensure_ascii=False)

    return generations


"""
Just printing all semantic entropy results
"""
def summarize_semantic_entropy_results(generations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not generations:
        return {
            "num_examples": 0,
            "mean_semantic_entropy": 0.0,
            "mean_normalized_semantic_entropy": 0.0,
            "mean_num_semantic_clusters": 0.0,
        }

    semantic_entropies = [row["semantic_entropy"]["semantic_entropy"] for row in generations]
    normalized_semantic_entropies = [
        row["semantic_entropy"]["normalized_semantic_entropy"] for row in generations
    ]
    cluster_counts = [row["semantic_entropy"]["num_semantic_clusters"] for row in generations]

    summary: Dict[str, Any] = {
        "num_examples": len(generations),
        "mean_semantic_entropy": sum(semantic_entropies) / len(semantic_entropies),
        "mean_normalized_semantic_entropy": sum(normalized_semantic_entropies) / len(normalized_semantic_entropies),
        "mean_num_semantic_clusters": sum(cluster_counts) / len(cluster_counts),
    }

    labels = [row.get("ground_truth_label") for row in generations if row.get("ground_truth_label") is not None]
    if labels:
        label_histogram: Dict[str, int] = {}
        for label in labels:
            key = str(label)
            label_histogram[key] = label_histogram.get(key, 0) + 1
        summary["ground_truth_label_histogram"] = label_histogram

    judged_rows = [row for row in generations if row.get("judge_label")]
    if judged_rows:
        judge_histogram: Dict[str, int] = {}
        exact_match_count = 0
        comparable_count = 0
        for row in judged_rows:
            judge_binary = int(row["judge_label"]["binary_label"])
            judge_key = str(judge_binary)
            judge_histogram[judge_key] = judge_histogram.get(judge_key, 0) + 1
            dataset_binary = _normalize_binary_dataset_label(row.get("ground_truth_label"))
            if dataset_binary is not None:
                comparable_count += 1
                if dataset_binary == judge_binary:
                    exact_match_count += 1
        summary["judge_label_histogram"] = judge_histogram
        if comparable_count:
            summary["judge_vs_dataset_label_accuracy"] = exact_match_count / comparable_count

    return summary


def generate_answers_resumable(
    model_ours: Union[str, AutoModelForCausalLM] = DEFAULT_SMALL_MODEL,
    dataset_name: str = "phantom",
    data_file: Optional[Union[str, Path]] = None,
    num_rows: int = DEFAULT_NUM_ROWS,
    k: int = DEFAULT_K,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
    judge_model: Optional[str] = DEFAULT_JUDGE_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    use_api_generation: bool = True,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    seed_start: int = 42,
    token_entropy_model: Optional[AutoModelForCausalLM] = None,
    token_entropy_tokenizer: Optional[AutoTokenizer] = None,
    evidence_model: Optional[AutoModelForSequenceClassification] = None,
    evidence_tokenizer: Optional[AutoTokenizer] = None,
    self_consistency_model: Optional[SentenceTransformer] = None,
) -> List[Dict[str, Any]]:
    """
    Resumable generation over PHANTOM subset.
    Saves outputs keyed by example_id and skips already processed examples.
    """

    _ensure_data_dirs()

    sanitized_model_name = str(model_ours).replace("/", "_").replace(":", "_")
    output_dir = get_output_dir_for_dataset(dataset_name)
    generations_path = output_dir / f"{sanitized_model_name}_{dataset_name}_generations.json"
    judge_labels_path = output_dir / f"{sanitized_model_name}_{dataset_name}_judge_labels.json"

    # Load existing outputs to resume
    if generations_path.exists():
        with generations_path.open("r", encoding="utf-8") as f:
            generations_dict = {str(row["example_id"]): row for row in json.load(f)}
    else:
        generations_dict = {}

    if judge_labels_path.exists():
        with judge_labels_path.open("r", encoding="utf-8") as f:
            judge_labels_dict = json.load(f)
    else:
        judge_labels_dict = {}

    dataset = load_dataset_for_pipeline(
        dataset_name=dataset_name,
        data_file=data_file,
        num_rows=num_rows,
    )

    if not use_api_generation:
        model, tokenizer, device = _load_model_and_tokenizer(
            model_ours=model_ours, tokenizer=tokenizer, device=device
        )


    # TOKEN ENTROPY STUFF
    if token_entropy_model is None or token_entropy_tokenizer is None:
        raise ValueError("token_entropy_model and token_entropy_tokenizer must be provided")
    
    if evidence_model is None or evidence_tokenizer is None:
        raise ValueError("evidence_model and evidence_tokenizer must be provided")
    
    if self_consistency_model is None:
        raise ValueError("self_consistency_model must be provided")
    
    #############################


    for idx, row in enumerate(dataset):
        example_id = str(row.get("id", row.get("Unnamed: 0", idx)))
        if (
            example_id in generations_dict
            and "semantic_entropy" in generations_dict[example_id]
            and "token_entropy" in generations_dict[example_id]
            and "evidence_consistency" in generations_dict[example_id]
            and "self_consistency" in generations_dict[example_id]
        ):
            print(f"[SKIP] example_id {example_id} already processed")
            continue

        print(f"[RUNNING] example_id {example_id} | Question preview: {row['query'][:50]}...")

        generation = generate_k_answers(
            model_ours=model_ours if use_api_generation else model,
            tokenizer=tokenizer,
            device=device,
            question=row["query"],
            context=row["context"],
            k=k,
            use_api_generation=use_api_generation,
            ollama_url=ollama_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed_start=seed_start
        )

        # Add example metadata
        generation["example_id"] = example_id
        generation["ground_truth_answer"] = row.get("answer")
        generation["ground_truth_label"] = row.get("ground_truth_label")
        generation["served_answer"] = generation["answers"][0] if generation["answers"] else ""

        # Save decoding configuration for reproducibility
        generation["decoding_config"] = {
            "model_name": str(model_ours),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "k": k,
            "seed_start": seed_start,
            "ollama_url": ollama_url,
        }

        # Compute semantic entropy
        semantic_log_probs = generation.get("total_log_probs")
        if semantic_log_probs and any(lp is None for lp in semantic_log_probs):
            semantic_log_probs = None
        generation["semantic_entropy"] = semantic_entropy(
            question=row["query"],
            k_sampled_answers=generation.get("answers", []),
            log_probs_for_token_probabilities=semantic_log_probs,
        )

        served_answer = generation["served_answer"].strip()

        # Compute token entropy
        served_answer = generation["served_answer"].strip()
        if served_answer:
            generation["token_entropy"] = token_entropy(
                tokenizer=token_entropy_tokenizer,
                model=token_entropy_model,
                answer=served_answer,
            )
        else:
            generation["token_entropy"] = None

        # Compute evidence consistency
        if served_answer:
            generation["evidence_consistency"] = evidence_consistency(
                tokenizer=evidence_tokenizer,
                model=evidence_model,
                context=row["context"],
                answer=served_answer,
            )
        else:
            generation["evidence_consistency"] = None


        # Compute self consistency
        generation["self_consistency"] = self_consistency(
            sampled_answers=generation.get("answers", []),
            sbert_model=self_consistency_model,
        )

        # Optional judge evaluation
        if judge_model:
            judge_output = generate_gt(
                big_model=judge_model,
                our_answer=generation["served_answer"],
                context=row["context"],
                question=row["query"],
                ollama_url=ollama_url,
            )
            generation["judge_label"] = judge_output
            judge_labels_dict[example_id] = judge_output

        # Save after each example for resumability
        generations_dict[example_id] = generation
        with generations_path.open("w", encoding="utf-8") as f:
            json.dump(list(generations_dict.values()), f, indent=2, ensure_ascii=False)
        with judge_labels_path.open("w", encoding="utf-8") as f:
            json.dump(judge_labels_dict, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Generations saved to {generations_path}")
    print(f"[DONE] Judge labels saved to {judge_labels_path}")

    return list(generations_dict.values())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PHANTOM semantic entropy generation on a small subset.")
    parser.add_argument("--model", default=DEFAULT_SMALL_MODEL, help="Generator model name.")
    parser.add_argument("--num-rows", type=int, default=DEFAULT_NUM_ROWS, help="Number of PHANTOM rows to process.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of sampled answers per example.")
    parser.add_argument("--device", default=None, help="Device override, for example cpu or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Maximum tokens per sampled answer.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--seed-start", type=int, default=1234)
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--no-judge", action="store_true", help="Skip API-based judging.")
    parser.add_argument("--local-generation", action="store_true", help="Use local Transformers generation instead of Ollama.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama chat API endpoint.")
    return parser


def main() -> None:
    _load_local_env()
    args = _build_arg_parser().parse_args()

    token_entropy_model, token_entropy_tokenizer, _ = _load_token_entropy_resources(
        model_name=DEFAULT_TOKEN_ENTROPY_MODEL,
        device=args.device,
    )

    evidence_model, evidence_tokenizer, _ = _load_evidence_consistency_resources(
        model_name=DEFAULT_NLI_MODEL,
        device=args.device,
    )

    self_consistency_model = _load_self_consistency_resources(
        model_name=DEFAULT_SELF_CONSISTENCY_MODEL,
        device=args.device,
    )

    

    phantom_generations = generate_answers_resumable(
        model_ours=args.model,
        dataset_name="phantom",
        data_file=DEFAULT_PHANTOM_LOCAL_FILE,
        num_rows=args.num_rows,
        k=args.k,
        device=args.device,
        judge_model=None if args.no_judge else args.judge_model,
        ollama_url=args.ollama_url,
        use_api_generation=not args.local_generation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed_start=args.seed_start,
        token_entropy_model=token_entropy_model,
        token_entropy_tokenizer=token_entropy_tokenizer,
        evidence_model=evidence_model,
        evidence_tokenizer=evidence_tokenizer,
        self_consistency_model=self_consistency_model,
    )

    wikiqa_generations = generate_answers_resumable(
        model_ours=args.model,
        dataset_name="wikiqa",
        data_file=DEFAULT_WIKIQA_LOCAL_FILE,
        num_rows=args.num_rows,
        k=args.k,
        device=args.device,
        judge_model=None if args.no_judge else args.judge_model,
        ollama_url=args.ollama_url,
        use_api_generation=not args.local_generation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed_start=args.seed_start,
        token_entropy_model=token_entropy_model,
        token_entropy_tokenizer=token_entropy_tokenizer,
        evidence_model=evidence_model,
        evidence_tokenizer=evidence_tokenizer,
        self_consistency_model=self_consistency_model,
    )

    phantom_summary = summarize_semantic_entropy_results(phantom_generations)
    wikiqa_summary = summarize_semantic_entropy_results(wikiqa_generations)

    phantom_summary_path = PHANTOM_OUTPUT_DIR / f"summary_{Path(DEFAULT_PHANTOM_LOCAL_FILE).stem}_first_{args.num_rows}_k{args.k}.json"
    wikiqa_summary_path = WIKIQA_OUTPUT_DIR / f"summary_{Path(DEFAULT_WIKIQA_LOCAL_FILE).stem}_first_{args.num_rows}_k{args.k}.json"

    with phantom_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(phantom_summary, handle, indent=2)

    with wikiqa_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(wikiqa_summary, handle, indent=2)

    print("\nPHANTOM summary:")
    print(json.dumps(phantom_summary, indent=2))
    print(f"PHANTOM detailed results saved to: {PHANTOM_OUTPUT_DIR}")
    print(f"PHANTOM summary saved to: {phantom_summary_path}")

    print("\nWikiQA summary:")
    print(json.dumps(wikiqa_summary, indent=2))
    print(f"WikiQA detailed results saved to: {WIKIQA_OUTPUT_DIR}")
    print(f"WikiQA summary saved to: {wikiqa_summary_path}")


if __name__ == "__main__":
    main()



