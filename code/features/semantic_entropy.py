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


_NLI_CACHE: Dict[str, Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]] = {}
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


"""
Setting device
"""
def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

"""
Basic preprocessing on the result. Removes empty spaces from end and beginning
"""
def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


"""
Given a model_id/name and tokenizer_id/name, loads it and returns them in a tuple
# TODO: Make the return signature consistent with load_mode_and_tokenizer function
"""
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


"""
TODO: Figure out what this does wrt to the pipeline
"""
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
