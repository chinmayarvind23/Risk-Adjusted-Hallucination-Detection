# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def token_entropy(tokenizer_name, model, answer):
#     """
#     - This is dependent on the tokenizer and therefore on the model too
#     - so, accept the model/tokenizer, and the sentence

#     - Input:
#         - Answer whose entropy we need to compute. Could be multiple sentences too. We just pass the entire string
#         - tokenizer
#         - model (optional, depending on how huggingface needs it)

#     - Output:
#         - the single entropy value
#     """
#     "return entropy value"
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     inputs = tokenizer(answer, return_tensors="pt")
#     ids = inputs.input_ids

#     device = next(model.parameters()).device
#     ids = ids.to(device)

#     with torch.no_grad():
#         outputs = model(ids)
#         logits = outputs.logits

#     shift_logits = logits[:, :-1, :]
#     shift_labels = ids[:, 1:]

#     probs = F.softmax(shift_logits, dim=-1)

#     log_probs = torch.log(probs + 1e-12)
#     entropy = -torch.sum(log_probs * probs, dim=-1)

#     if tokenizer.pad_token_id is not None:
#         mask = (shift_labels != tokenizer.pad_token_id).float()
#         entropy = entropy * mask
#         avg_entropy = entropy.sum() / mask.sum()
#     else:
#         avg_entropy = entropy.mean()

#     return avg_entropy.item()


import torch
import torch.nn.functional as F

def token_entropy(tokenizer, model, answer):
    """
    Compute average token entropy over the answer string using a
    causal LM and a tokenizer that are already loaded.
    """
    inputs = tokenizer(answer, return_tensors="pt")
    ids = inputs.input_ids

    device = next(model.parameters()).device
    ids = ids.to(device)

    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]

    probs = F.softmax(shift_logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(log_probs * probs, dim=-1)

    if tokenizer.pad_token_id is not None:
        mask = (shift_labels != tokenizer.pad_token_id).float()
        entropy = entropy * mask
        avg_entropy = entropy.sum() / mask.sum()
    else:
        avg_entropy = entropy.mean()

    return avg_entropy.item()