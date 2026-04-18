# PHANTOM Calibration and Abstention Findings

## What this file is about

The overall goal is:

1. Use model features to estimate how likely an answer is to be unsupported or hallucinatory.
2. Calibrate that score so the probability is easier to trust.
3. Add an abstention rule so the system can refuse risky cases instead of answering everything.

In this report, a higher risk score means the answer is more likely to be unsupported.

## What the detector is using

The detector uses these four features:

- `mean_token_nll`
- `self_consistency_disagreement`
- `semantic_entropy`
- `groundedness_score`

### Simple meaning of each feature

- `mean_token_nll`: how surprised the model was while generating the answer. More surprise usually means more risk.
- `self_consistency_disagreement`: how much multiple generations disagree with each other. More disagreement usually means more risk.
- `semantic_entropy`: how semantically spread out the answers are. More spread usually means more risk.
- `groundedness_score`: how well the answer seems supported by evidence. Higher groundedness usually means less risk.

## What the learned weights mean

The tuned detector is a logistic regression model with these weights:

- `mean_token_nll`: `0.3014`
- `self_consistency_disagreement`: `0.7280`
- `semantic_entropy`: `0.2007`
- `groundedness_score`: `-0.3692`
- intercept: `-0.9983`

### Interpretation

- Positive weights increase predicted hallucination risk.
- Negative weights decrease predicted hallucination risk.

So for this model:

- More disagreement is the strongest warning sign.
- More token-level uncertainty also increases risk.
- More semantic entropy also increases risk.
- More groundedness lowers risk.

This matches the intended behavior of the features.

## What calibration is doing

The raw detector already ranks examples reasonably well, but raw model probabilities are not always well calibrated.

Calibration tries to make the predicted probability better match reality.

Example:

- If the model says "0.80 risk" on many examples, then roughly 80 percent of those examples should really be unsupported.

This report uses **Platt scaling** for calibration.

### Why Platt scaling was used

Platt scaling fits a simple logistic transformation on top of the detector score.

It does not change the ordering of examples very much, but it can make the probabilities more usable.

The fitted Platt parameters are:

- coefficient: `1.4564`
- intercept: `0.5817`

## How to read the metrics

### Calibration metrics

- `NLL`: lower is better. Penalizes confident wrong predictions.
- `ECE`: lower is better. Measures how far predicted probabilities are from actual outcomes.
- `Brier`: lower is better. Measures probability error.

### Classification metrics

These are computed after turning probabilities into labels using threshold `0.5`.

- `AUROC`: higher is better. Measures ranking quality.
- `AUPRC`: higher is better. Useful when the positive class is important.
- `accuracy`: overall fraction correct.
- `precision`: when the detector says "unsupported," how often it is right.
- `recall`: of all truly unsupported answers, how many it catches.
- `F1`: balance between precision and recall.
- `positive_rate_at_threshold`: how often the model predicts the positive class at threshold `0.5`.

Here, the positive class means **unsupported / hallucinated**.

## Validation results before and after calibration

### Raw validation metrics

- NLL: `0.4677`
- ECE: `0.0694`
- Brier: `0.1514`
- AUROC: `0.8331`
- AUPRC: `0.7555`
- Accuracy: `0.7807`
- Precision: `0.8333`
- Recall: `0.4082`
- F1: `0.5479`

### Calibrated validation metrics

- NLL: `0.4529`
- ECE: `0.0411`
- Brier: `0.1472`
- AUROC: `0.8331`
- AUPRC: `0.7555`
- Accuracy: `0.7874`
- Precision: `0.7742`
- Recall: `0.4898`
- F1: `0.6000`

### What changed on validation

- Calibration improved all three probability-quality metrics: NLL, ECE, and Brier.
- AUROC and AUPRC stayed the same.
- Accuracy improved a little.
- Recall improved from `0.4082` to `0.4898`.
- F1 improved from `0.5479` to `0.6000`.
- Precision dropped a little because the calibrated model calls more cases risky.

This is a reasonable tradeoff. The calibrated model catches more unsupported answers, even if it becomes slightly less conservative.

## Test results before and after calibration

### Raw test metrics

- NLL: `0.5207`
- ECE: `0.0598`
- Brier: `0.1695`
- AUROC: `0.7730`
- AUPRC: `0.6451`
- Accuracy: `0.7591`
- Precision: `0.7674`
- Recall: `0.3438`
- F1: `0.4748`

### Calibrated test metrics

- NLL: `0.5248`
- ECE: `0.0734`
- Brier: `0.1652`
- AUROC: `0.7730`
- AUPRC: `0.6451`
- Accuracy: `0.7855`
- Precision: `0.7818`
- Recall: `0.4479`
- F1: `0.5695`

### What changed on test

- Brier score improved.
- NLL and ECE got a little worse.
- AUROC and AUPRC stayed the same.
- Accuracy improved.
- Precision improved slightly.
- Recall improved a lot, from `0.3438` to `0.4479`.
- F1 improved a lot, from `0.4748` to `0.5695`.

### Bottom line on calibration

Calibration gave **mixed probability-quality results on test**, but it gave **better thresholded decision quality** at `0.5`, especially better recall and F1.

That means the calibrated score is more useful for making decisions, even if it is not uniformly better on every calibration metric.

## Why AUROC and AUPRC did not change

Platt scaling is a monotonic transformation. That means it changes the scale of the scores, but it mostly keeps the ranking order of examples the same.

Because AUROC and AUPRC depend on ranking, not on the exact probability values, they stay unchanged.

## What abstention means here

Abstention means the system is allowed to say:

- "I will answer this"
- or "this looks too risky, so I will abstain"

This is useful because some examples are clearly riskier than others. If we refuse the riskiest cases, the answers we keep should be more reliable.

## The abstention rule used here

The JSON says:

> Choose the validation risk threshold that maximizes selective accuracy subject to the minimum coverage constraint.

In simple words, that means:

1. Try many possible risk cutoffs.
2. Only keep cutoffs where the system still answers at least 80 percent of cases.
3. Among those, choose the cutoff that makes the kept answers as accurate as possible.

The minimum coverage constraint is:

- `0.8`

So the system must still answer at least 80 percent of examples.

## Chosen abstention threshold

The selected validation threshold is:

- `0.5132`

Interpretation:

- If calibrated risk is **above** `0.5132`, abstain.
- If calibrated risk is **at or below** `0.5132`, keep the answer.

## Selected validation operating point

At the chosen threshold on validation:

- coverage: `0.8040`
- abstention rate: `0.1960`
- selective risk: `0.2107`
- selective accuracy: `0.7893`

### What those numbers mean

- `coverage = 0.8040`: the system answers about 80.4 percent of examples.
- `abstention_rate = 0.1960`: the system refuses about 19.6 percent of examples.
- `selective_risk = 0.2107`: among the answers it keeps, about 21.1 percent are still wrong / unsupported.
- `selective_accuracy = 0.7893`: among the answers it keeps, about 78.9 percent are correct / supported.

## Test operating point at the frozen threshold

When that same decision rule is carried to test, the reported operating point is:

- threshold: `0.4986`
- coverage: `0.8185`
- abstention rate: `0.1815`
- selective risk: `0.2137`
- selective accuracy: `0.7863`

### Interpretation

On test, the abstention policy is very close to validation:

- It still answers more than 80 percent of cases.
- It abstains on about 18.2 percent.
- The kept answers are about 78.6 percent accurate.

This is a good sign. The abstention rule transfers reasonably well from validation to test.

## Why the test threshold number is slightly different

The frozen validation threshold is `0.5131736523778971`, but the test operating point is shown with threshold `0.49861409854048155`.

This usually happens because the test curve is built from the set of actual score values seen on the test set. The reported point is the nearest achievable point on that discrete curve.

So the policy is still the same idea. The exact printed threshold differs because the score values on test are different.

## What the long validation and test curves mean

The large `validation_curve` and `test_curve` sections show what happens as you move the abstention threshold.

General pattern:

- When you abstain on more examples, coverage goes down.
- As coverage goes down, selective accuracy usually goes up.
- If you force the system to answer everything, selective accuracy drops.

This is the usual tradeoff:

- More answers means lower quality.
- Fewer answers means higher quality.

## The most important comparison

### If the system answers everything

At full coverage on validation:

- selective accuracy: `0.6744`
- selective risk: `0.3256`

At full coverage on test:

- selective accuracy: `0.6832`
- selective risk: `0.3168`

### If the system uses abstention

At the selected operating point on validation:

- coverage: `0.8040`
- selective accuracy: `0.7893`
- selective risk: `0.2107`

At the selected operating point on test:

- coverage: `0.8185`
- selective accuracy: `0.7863`
- selective risk: `0.2137`

### Simple takeaway

By refusing about 18 to 20 percent of the riskiest cases, the system improves the quality of the answers it keeps:

- Test selective accuracy rises from about `68.3%` to about `78.6%`.
- Test selective risk falls from about `31.7%` to about `21.4%`.

That is the main practical win in this report.

## Overall conclusion

This JSON shows three things:

1. The detector is learning sensible signals.
   Higher disagreement, uncertainty, and entropy increase risk, while groundedness lowers it.
2. Platt scaling is a reasonable calibration choice.
   It does not improve every probability metric on test, but it improves decision quality at the default threshold and makes the score useful for abstention.
3. The abstention policy is effective.
   By abstaining on the riskiest roughly 20 percent of cases, the system keeps more reliable answers while still covering about 80 percent of the dataset.

## Short version

> The PHANTOM detector can rank risky answers fairly well, Platt scaling makes the score usable for decision-making, and abstaining on the riskiest 20 percent of cases gives a clear quality boost while still answering most questions.
