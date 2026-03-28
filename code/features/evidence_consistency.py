

#TODO: Fix this. Improve implementation (if needed) and work on the required imports
import torch

def evidence_consistency(tokenizer, model, context, answer):
    """
    Compute NLI-based evidence consistency scores between context and answer.
    Returns class probabilities and a scalar consistency score.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        context,
        answer,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits[0], dim=-1)

    id2label = model.config.id2label
    prediction = {
        id2label[i].lower(): float(probs[i].item())
        for i in range(len(probs))
    }

    entailment = prediction.get("entailment", 0.0)
    contradiction = prediction.get("contradiction", 0.0)

    return {
        "scores": prediction,
        "consistency_score": entailment - contradiction,
    }