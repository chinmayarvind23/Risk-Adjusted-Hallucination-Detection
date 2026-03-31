from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def token_entropy(
    token_logprobs: Optional[Sequence[float]] = None,
    token_details: Optional[Sequence[Dict[str, Any]]] = None,
    tokenizer=None,
    model=None,
    answer: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute token-level uncertainty for the served answer.

    Preferred path:
    - Use generation-time token logprobs returned by the generator backend.

    Fallback path:
    - If a local causal LM, tokenizer, and answer are supplied, compute the
      next-token distribution entropy from logits and derive token logprobs for
      the realized answer tokens.
    """
    if token_logprobs:
        valid_logprobs = [float(lp) for lp in token_logprobs if lp is not None]
        if valid_logprobs:
            mean_logprob = sum(valid_logprobs) / len(valid_logprobs)
            total_logprob = sum(valid_logprobs)

            entropy_values: List[float] = []
            if token_details:
                for token in token_details:
                    entropy_value = _safe_float(token.get("entropy"))
                    if entropy_value is not None:
                        entropy_values.append(entropy_value)

            return {
                "mean_token_nll": -mean_logprob,
                "mean_token_logprob": mean_logprob,
                "sum_token_logprob": total_logprob,
                "num_scored_tokens": len(valid_logprobs),
                "mean_next_token_entropy": (
                    sum(entropy_values) / len(entropy_values) if entropy_values else None
                ),
                "source": "generator_logprobs",
            }

    if tokenizer is None or model is None or not answer:
        return {
            "mean_token_nll": None,
            "mean_token_logprob": None,
            "sum_token_logprob": None,
            "num_scored_tokens": 0,
            "mean_next_token_entropy": None,
            "source": "unavailable",
        }

    inputs = tokenizer(answer, return_tensors="pt")
    input_ids = inputs.input_ids

    if input_ids.shape[-1] <= 1:
        return {
            "mean_token_nll": None,
            "mean_token_logprob": None,
            "sum_token_logprob": None,
            "num_scored_tokens": 0,
            "mean_next_token_entropy": None,
            "source": "local_logits",
        }

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]

    target_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_logprob_tensor = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_entropy_tensor = -(probs * log_probs).sum(dim=-1)

    mean_logprob = float(token_logprob_tensor.mean().item())
    total_logprob = float(token_logprob_tensor.sum().item())
    mean_entropy = float(token_entropy_tensor.mean().item())

    return {
        "mean_token_nll": -mean_logprob,
        "mean_token_logprob": mean_logprob,
        "sum_token_logprob": total_logprob,
        "num_scored_tokens": int(token_logprob_tensor.numel()),
        "mean_next_token_entropy": mean_entropy,
        "source": "local_logits",
    }
