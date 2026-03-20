from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


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

def _ensure_data_dirs() -> None:
    PHANTOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


_NLI_CACHE: Dict[str, Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]] = {}


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

def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _load_nli_model_and_tokenizer(
    model_name: str = DEFAULT_NLI_MODEL,
    device: Optional[str] = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
    cache_key = f"{model_name}::{device or 'auto'}"
    if cache_key in _NLI_CACHE:
        return _NLI_CACHE[cache_key]

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

    loaded = (model, tokenizer, resolved_device)
    _NLI_CACHE[cache_key] = loaded
    return loaded


def _extract_label_indices(model: AutoModelForSequenceClassification) -> Tuple[int, Optional[int]]:
    entailment_idx: Optional[int] = None
    contradiction_idx: Optional[int] = None

    for idx, label in model.config.id2label.items():
        normalized = label.lower()
        if "entail" in normalized:
            entailment_idx = int(idx)
        elif "contradict" in normalized:
            contradiction_idx = int(idx)

    if entailment_idx is None:
        raise ValueError(f"Could not find an entailment label in {model.config.id2label}")

    return entailment_idx, contradiction_idx


def _format_answer_for_entailment(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer.strip()}"


def _predict_nli_probs(
    premise: str,
    hypothesis: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
) -> torch.Tensor:
    encoded = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    return torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()


def _bidirectional_entailment(
    question: str,
    answer_a: str,
    answer_b: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    entailment_threshold: float,
) -> Tuple[bool, Dict[str, float]]:
    if _normalize_answer(answer_a) == _normalize_answer(answer_b):
        return True, {"forward_entailment": 1.0, "backward_entailment": 1.0}

    entailment_idx, _ = _extract_label_indices(model)
    premise_a = _format_answer_for_entailment(question, answer_a)
    premise_b = _format_answer_for_entailment(question, answer_b)

    forward_probs = _predict_nli_probs(
        premise=premise_a,
        hypothesis=premise_b,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    backward_probs = _predict_nli_probs(
        premise=premise_b,
        hypothesis=premise_a,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    forward_entailment = float(forward_probs[entailment_idx].item())
    backward_entailment = float(backward_probs[entailment_idx].item())
    equivalent = (
        forward_entailment >= entailment_threshold
        and backward_entailment >= entailment_threshold
    )

    return equivalent, {
        "forward_entailment": forward_entailment,
        "backward_entailment": backward_entailment,
    }


def _cluster_semantic_variants(
    question: str,
    answers: Sequence[str],
    entailment_model_name: str = DEFAULT_NLI_MODEL,
    entailment_model: Optional[AutoModelForSequenceClassification] = None,
    entailment_tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
    entailment_threshold: float = 0.5,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    if entailment_model is None or entailment_tokenizer is None:
        entailment_model, entailment_tokenizer, resolved_device = _load_nli_model_and_tokenizer(
            model_name=entailment_model_name,
            device=device,
        )
    else:
        resolved_device = _resolve_device(device)

    clusters: List[List[int]] = []
    comparisons: List[Dict[str, Any]] = []

    for idx, answer in enumerate(answers):
        placed = False
        for cluster in clusters:
            representative_idx = cluster[0]
            equivalent, scores = _bidirectional_entailment(
                question=question,
                answer_a=answer,
                answer_b=answers[representative_idx],
                model=entailment_model,
                tokenizer=entailment_tokenizer,
                device=resolved_device,
                entailment_threshold=entailment_threshold,
            )
            comparisons.append(
                {
                    "candidate_index": idx,
                    "representative_index": representative_idx,
                    "equivalent": equivalent,
                    **scores,
                }
            )
            if equivalent:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    return clusters, comparisons


def build_phantom_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the provided context. "
        "If the context does not support an answer, say that the answer is not supported.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


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


def _validate_nonempty_answers(answers: Sequence[str], model_name: str) -> None:
    if answers and all(not answer.strip() for answer in answers):
        raise ValueError(
            f"{model_name} returned empty final answers after cleaning. "
            "This usually means the model is still emitting thinking text instead of final content."
        )


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


def semantic_entropy(
    question: str,
    k_sampled_answers: Sequence[str],
    log_probs_for_token_probabilities: Optional[Sequence[float]] = None,
    entailment_model_name: str = DEFAULT_NLI_MODEL,
    entailment_model: Optional[AutoModelForSequenceClassification] = None,
    entailment_tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = None,
    entailment_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Semantic entropy estimator following Farquhar et al. (Nature, 2024):
    1. Cluster sampled answers by bidirectional entailment (if A true makes B true, then A entails B).
    2. Sum sequence probabilities within each semantic cluster.
    3. Normalize cluster masses across observed clusters.
    4. Compute entropy over the semantic-cluster distribution.

    If sequence log-probabilities are not supplied, this falls back to the
    paper's discrete semantic entropy estimator using cluster frequencies.
    """
    if not k_sampled_answers:
        raise ValueError("k_sampled_answers must contain at least one sampled answer")

    if (
        log_probs_for_token_probabilities is not None
        and len(k_sampled_answers) != len(log_probs_for_token_probabilities)
    ):
        raise ValueError("k_sampled_answers and log_probs_for_token_probabilities must have the same length")

    clusters, comparisons = _cluster_semantic_variants(
        question=question,
        answers=k_sampled_answers,
        entailment_model_name=entailment_model_name,
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        device=device,
        entailment_threshold=entailment_threshold,
    )

    cluster_answers: List[List[str]] = [[k_sampled_answers[idx] for idx in cluster] for cluster in clusters]

    if log_probs_for_token_probabilities is None:
        cluster_probabilities = [len(cluster) / len(k_sampled_answers) for cluster in clusters]
        estimator = "discrete"
        raw_cluster_masses = cluster_probabilities[:]
        cluster_log_masses = [math.log(probability) for probability in cluster_probabilities]
    else:
        answer_log_probs = torch.tensor(log_probs_for_token_probabilities, dtype=torch.float64)
        cluster_log_masses = [
            float(torch.logsumexp(answer_log_probs[cluster], dim=0).item())
            for cluster in clusters
        ]
        normalization_log_mass = float(torch.logsumexp(torch.tensor(cluster_log_masses, dtype=torch.float64), dim=0).item())
        cluster_probabilities = [
            math.exp(cluster_log_mass - normalization_log_mass)
            for cluster_log_mass in cluster_log_masses
        ]
        raw_cluster_masses = [math.exp(cluster_log_mass) for cluster_log_mass in cluster_log_masses]
        estimator = "rao_blackwellized"

    entropy = 0.0
    for cluster_probability in cluster_probabilities:
        if cluster_probability > 0:
            entropy -= cluster_probability * math.log(cluster_probability)

    max_entropy = math.log(len(cluster_probabilities)) if len(cluster_probabilities) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "semantic_entropy": entropy,
        "normalized_semantic_entropy": normalized_entropy,
        "num_semantic_clusters": len(clusters),
        "estimator": estimator,
        "entailment_model_name": entailment_model_name,
        "entailment_threshold": entailment_threshold,
        "cluster_log_masses": cluster_log_masses,
        "raw_cluster_masses": raw_cluster_masses,
        "cluster_probabilities": cluster_probabilities,
        "clusters": cluster_answers,
        "cluster_indices": clusters,
        "pairwise_comparisons": comparisons,
    }


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


def generate_answers_for_phantom_subset(
    model_ours: Union[str, AutoModelForCausalLM] = DEFAULT_SMALL_MODEL,
    data_file: str = DEFAULT_PHANTOM_FILE,
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
    dataset = load_phantom_dataset(data_file=data_file, num_rows=num_rows)

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
        example_id = row.get("id", idx)
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
    data_file: str = DEFAULT_PHANTOM_FILE,
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
) -> List[Dict[str, Any]]:
    """
    Resumable generation over PHANTOM subset.
    Saves outputs keyed by example_id and skips already processed examples.
    """

    _ensure_data_dirs()

    sanitized_model_name = str(model_ours).replace("/", "_").replace(":", "_")
    generations_path = PHANTOM_OUTPUT_DIR / f"{sanitized_model_name}_generations.json"
    judge_labels_path = PHANTOM_OUTPUT_DIR / f"{sanitized_model_name}_judge_labels.json"

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

    dataset = load_phantom_dataset(data_file=data_file, num_rows=num_rows)

    if not use_api_generation:
        model, tokenizer, device = _load_model_and_tokenizer(
            model_ours=model_ours, tokenizer=tokenizer, device=device
        )

    for idx, row in enumerate(dataset):
        example_id = str(row.get("id", idx))
        if example_id in generations_dict:
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
    parser.add_argument("--data-file", default=DEFAULT_PHANTOM_FILE, help="PHANTOM dataset file inside the HF repo.")
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

    # generations = generate_answers_for_phantom_subset(
    #     model_ours=args.model,
    #     data_file=args.data_file,
    #     num_rows=args.num_rows,
    #     k=args.k,
    #     device=args.device,
    #     judge_model=None if args.no_judge else args.judge_model,
    #     ollama_url=args.ollama_url,
    #     use_api_generation=not args.local_generation,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     top_k=args.top_k,
    # )

    generations = generate_answers_resumable(
        model_ours=args.model,
        data_file=args.data_file,
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
    )
    summary = summarize_semantic_entropy_results(generations)

    summary_path = PHANTOM_OUTPUT_DIR / f"summary_{Path(args.data_file).stem}_first_{args.num_rows}_k{args.k}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Detailed results saved to: {PHANTOM_OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
