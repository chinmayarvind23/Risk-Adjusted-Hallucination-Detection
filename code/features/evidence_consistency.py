from __future__ import annotations

"""
Compute groundedness by checking answer sentences against retrieved evidence.

This is an evidence consistency feature. It breaks an answer into sentences,
compares each sentence against chunks of the provided context with NLI, and
aggregates the strongest support signal for each sentence.

Higher groundedness means the answer looks better supported by the available
evidence. Lower groundedness means the answer is more weakly supported or
contradicted.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import nltk
import torch


def _sent_tokenize(text: str) -> List[str]:
    """Split text into sentences, with a simple fallback when punkt is missing."""
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            pass

    fallback = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]
    return [f"{segment}." if not segment.endswith((".", "!", "?")) else segment for segment in fallback]


def _extract_label_indices(model) -> Tuple[int, Optional[int], Optional[int]]:
    """Read the entailment, contradiction, and neutral label ids from the NLI model."""
    entailment_idx: Optional[int] = None
    contradiction_idx: Optional[int] = None
    neutral_idx: Optional[int] = None

    for idx, label in model.config.id2label.items():
        normalized = str(label).lower()
        if "entail" in normalized:
            entailment_idx = int(idx)
        elif "contradict" in normalized:
            contradiction_idx = int(idx)
        elif "neutral" in normalized:
            neutral_idx = int(idx)

    if entailment_idx is None:
        raise ValueError(f"Could not find entailment label in {model.config.id2label}")

    return entailment_idx, contradiction_idx, neutral_idx


def _chunk_evidence(text: str, max_chars: int = 1200) -> List[str]:
    """Chunk long evidence text into smaller spans for NLI scoring."""
    sentences = _sent_tokenize(text)
    if not sentences:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _predict_probs(tokenizer, model, premise: str, hypothesis: str) -> torch.Tensor:
    """Run one NLI forward pass and return class probabilities."""
    device = next(model.parameters()).device
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


def evidence_consistency(tokenizer, model, context: str, answer: str) -> Dict[str, Any]:
    """
    Sentence-level evidence groundedness.

    For each answer sentence, score it against evidence chunks with NLI and keep
    the chunk with the strongest entailment signal. Aggregate the selected
    sentence-level judgments into answer-level groundedness statistics.
    """
    answer_sentences = _sent_tokenize(answer)
    evidence_chunks = _chunk_evidence(context)

    if not answer_sentences:
        return {
            "groundedness_score": None,
            "entailed_fraction": 0.0,
            "contradicted_fraction": 0.0,
            "neutral_fraction": 0.0,
            "mean_entailment": 0.0,
            "mean_contradiction": 0.0,
            "mean_neutral": 0.0,
            "sentence_results": [],
        }

    if not evidence_chunks:
        sentence_results = []
        for sentence in answer_sentences:
            sentence_results.append(
                {
                    "sentence": sentence,
                    "best_chunk_index": None,
                    "best_chunk_text": "",
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "label": "neutral",
                }
            )
        return {
            "groundedness_score": 0.0,
            "entailed_fraction": 0.0,
            "contradicted_fraction": 0.0,
            "neutral_fraction": 1.0,
            "mean_entailment": 0.0,
            "mean_contradiction": 0.0,
            "mean_neutral": 1.0,
            "sentence_results": sentence_results,
        }

    entailment_idx, contradiction_idx, neutral_idx = _extract_label_indices(model)
    sentence_results: List[Dict[str, Any]] = []

    # Each answer sentence is matched against all evidence chunks. We keep the
    # chunk with the best entailment-minus-contradiction score.
    for sentence in answer_sentences:
        best_result: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for chunk_index, chunk in enumerate(evidence_chunks):
            probs = _predict_probs(
                tokenizer=tokenizer,
                model=model,
                premise=chunk,
                hypothesis=sentence,
            )

            entailment = float(probs[entailment_idx].item())
            contradiction = float(probs[contradiction_idx].item()) if contradiction_idx is not None else 0.0
            neutral = float(probs[neutral_idx].item()) if neutral_idx is not None else max(0.0, 1.0 - entailment - contradiction)

            # Prefer support while penalizing contradiction when choosing the
            # evidence chunk representing the sentence.
            support_score = entailment - contradiction
            if support_score > best_score:
                best_score = support_score
                label = "entailment"
                if contradiction > max(entailment, neutral):
                    label = "contradiction"
                elif neutral > max(entailment, contradiction):
                    label = "neutral"

                best_result = {
                    "sentence": sentence,
                    "best_chunk_index": chunk_index,
                    "best_chunk_text": chunk,
                    "entailment": entailment,
                    "contradiction": contradiction,
                    "neutral": neutral,
                    "label": label,
                }

        if best_result is not None:
            sentence_results.append(best_result)

    if not sentence_results:
        return {
            "groundedness_score": None,
            "entailed_fraction": 0.0,
            "contradicted_fraction": 0.0,
            "neutral_fraction": 0.0,
            "mean_entailment": 0.0,
            "mean_contradiction": 0.0,
            "mean_neutral": 0.0,
            "sentence_results": [],
        }

    total_sentences = len(sentence_results)
    entailed_count = sum(result["label"] == "entailment" for result in sentence_results)
    contradicted_count = sum(result["label"] == "contradiction" for result in sentence_results)
    neutral_count = sum(result["label"] == "neutral" for result in sentence_results)

    entailed_fraction = entailed_count / total_sentences
    contradicted_fraction = contradicted_count / total_sentences
    neutral_fraction = neutral_count / total_sentences

    mean_entailment = sum(result["entailment"] for result in sentence_results) / total_sentences
    mean_contradiction = sum(result["contradiction"] for result in sentence_results) / total_sentences
    mean_neutral = sum(result["neutral"] for result in sentence_results) / total_sentences
    # The continuous score is the main detector feature because it preserves
    # more information than a hard sentence label alone.
    discrete_groundedness_score = entailed_fraction - contradicted_fraction
    continuous_groundedness_score = mean_entailment - mean_contradiction

    return {
        "groundedness_score": continuous_groundedness_score,
        "discrete_groundedness_score": discrete_groundedness_score,
        "entailed_fraction": entailed_fraction,
        "contradicted_fraction": contradicted_fraction,
        "neutral_fraction": neutral_fraction,
        "mean_entailment": mean_entailment,
        "mean_contradiction": mean_contradiction,
        "mean_neutral": mean_neutral,
        "sentence_results": sentence_results,
    }
