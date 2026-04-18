**What the tuned detector is doing**
- It is using the 4 features together to predict whether a generated answer is unsupported.
- The strongest signal is `self_consistency_disagreement`.
- `groundedness_score` helps in the opposite direction: more grounded answers are less likely to be hallucinations.
- `mean_token_nll` and `semantic_entropy` also help, but less strongly.

**What the coefficients mean**
- `self_consistency_disagreement = +0.728`
  - when sampled answers disagree more, hallucination risk goes up the most
- `groundedness_score = -0.369`
  - when evidence support is stronger, hallucination risk goes down
- `mean_token_nll = +0.301`
  - when token-level uncertainty is higher, hallucination risk goes up
- `semantic_entropy = +0.201`
  - when meaning-level uncertainty is higher, hallucination risk also goes up

So the model learned the expected pattern:
- more uncertainty => more risk
- more groundedness => less risk

**How good the tuned model is**
On the **validation set**:
- `AUROC = 0.833`
- `AUPRC = 0.755`
- `accuracy = 0.774`
- `precision = 0.636`
- `recall = 0.714`
- `F1 = 0.673`

On the **test set**:
- `AUROC = 0.773`
- `AUPRC = 0.645`
- `accuracy = 0.759`
- `precision = 0.610`
- `recall = 0.667`
- `F1 = 0.637`

Simple interpretation:
- the model separates supported vs unsupported answers fairly well
- it is much better after tuning the threshold
- it now catches many more hallucinations than the default threshold version
- the learned model performs better overall than the manual weighted baseline

**What tuning changed**
- the best model still used logistic regression
- it did **not** need class balancing
- the biggest improvement came from lowering the decision threshold from `0.5` to about `0.269`

That means:
- the original model was too conservative
- once you lower the threshold, recall and F1 improve a lot

This is evidence that the 4-feature approach works because:
- the combined detector performs meaningfully better than the manual weighting baseline
- the learned coefficients match the intended direction of the features
- all four signals contribute, especially self-consistency and groundedness