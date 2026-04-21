# Calibration and Abstention Findings

## Canonical artifacts

This folder should use these files as the final PHANTOM calibration and abstention artifacts:

- `phantom_4000_calibration_abstention_report.json`: full machine-readable results
- `calibration_findings.md`: human-readable summary
- `phantom_4000_reliability_diagram_val.png`
- `phantom_4000_reliability_diagram_test.png`
- `phantom_4000_risk_coverage_curve.png`
- `phantom_4000_accuracy_coverage_curve.png`

## What was done

After training the tuned logistic regression detector, the next step was to calibrate its risk scores and turn them into an abstention rule.

The final workflow was:

1. Use the tuned PHANTOM detector.
2. Fit **Platt scaling** on the PHANTOM validation split.
3. Convert detector outputs into calibrated unsupported-risk scores.
4. Evaluate raw and calibrated scores on validation and test.
5. Choose an abstention threshold on validation subject to the minimum coverage constraint.
6. Freeze that threshold.
7. Evaluate the frozen rule on test.

## Detector and calibration parameters

### Detector coefficients

- `mean_token_nll`: `0.3014`
- `self_consistency_disagreement`: `0.7280`
- `semantic_entropy`: `0.2007`
- `groundedness_score`: `-0.3692`
- intercept: `-0.9983`

Interpretation:

- Higher token uncertainty, self-disagreement, and semantic entropy increase hallucination risk.
- Higher groundedness lowers hallucination risk.
- Self-consistency disagreement is the strongest positive warning signal.

### Platt scaling parameters

- coefficient: `1.4564`
- intercept: `0.5817`

The raw detector produces a logit:

`z = b + w1*x1 + w2*x2 + w3*x3 + w4*x4`

The raw risk score is:

`p_raw = 1 / (1 + exp(-z))`

Platt scaling then converts that score into a calibrated probability:

`p_calibrated = 1 / (1 + exp(-(a*z + c)))`

## Label counts

Validation labels:

- supported class `0`: `203`
- unsupported class `1`: `98`

Test labels:

- supported class `0`: `207`
- unsupported class `1`: `96`

## Short summary tables

### Raw vs calibrated metrics

| Split | Setting | NLL | ECE | Brier | AUROC | AUPRC | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Raw | 0.4677 | 0.0694 | 0.1514 | 0.8331 | 0.7555 | 0.7807 | 0.8333 | 0.4082 | 0.5479 |
| Validation | Calibrated | 0.4529 | 0.0411 | 0.1472 | 0.8331 | 0.7555 | 0.7874 | 0.7742 | 0.4898 | 0.6000 |
| Test | Raw | 0.5207 | 0.0598 | 0.1695 | 0.7730 | 0.6451 | 0.7591 | 0.7674 | 0.3438 | 0.4748 |
| Test | Calibrated | 0.5248 | 0.0734 | 0.1652 | 0.7730 | 0.6451 | 0.7855 | 0.7818 | 0.4479 | 0.5695 |

### Full coverage vs selected abstention operating point

| Split | Operating point | Coverage | Abstention rate | Selective risk | Selective accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Full coverage | 1.0000 | 0.0000 | 0.3256 | 0.6744 |
| Validation | Selected operating point | 0.8040 | 0.1960 | 0.2107 | 0.7893 |
| Test | Full coverage | 1.0000 | 0.0000 | 0.3168 | 0.6832 |
| Test | Frozen-threshold test point | 0.8185 | 0.1815 | 0.2137 | 0.7863 |

## How to read the calibration metrics

- `NLL`: lower is better. It heavily penalizes confident wrong predictions.
- `ECE`: lower is better. It measures how close predicted probabilities are to actual outcomes.
- `Brier`: lower is better. It measures squared probability error.

## How to read the classification metrics

These are computed using the fixed classification threshold `0.5`.

- `AUROC`: ranking quality across thresholds
- `AUPRC`: precision-recall ranking quality
- `accuracy`: total fraction correct
- `precision`: among predicted unsupported answers, fraction truly unsupported
- `recall`: among truly unsupported answers, fraction detected
- `F1`: balance between precision and recall

The positive class is **unsupported / hallucinated**.

## Reliability diagram

Files:

- `phantom_4000_reliability_diagram_val.png`
- `phantom_4000_reliability_diagram_test.png`

What this plot means:

- x-axis: predicted unsupported risk
- y-axis: actual unsupported rate
- diagonal line: perfect calibration

How to read it:

- points near the diagonal mean the risk score is well calibrated
- points below the diagonal mean the model predicts too much risk
- points above the diagonal mean the model predicts too little risk

What happened here:

- On validation, Platt scaling improved all three probability-quality metrics.
- Validation ECE improved from `0.0694` to `0.0411`.
- Validation Brier improved from `0.1514` to `0.1472`.
- Validation NLL improved from `0.4677` to `0.4529`.
- On test, calibration was mixed.
- Test Brier improved from `0.1695` to `0.1652`.
- Test ECE worsened from `0.0598` to `0.0734`.
- Test NLL worsened from `0.5207` to `0.5248`.

Interpretation:

- The detector was already a reasonable risk ranker before calibration.
- Platt scaling improved in-domain probability quality on validation.
- Out-of-sample probability calibration on test is mixed rather than uniformly better.

## Why AUROC and AUPRC do not change

Platt scaling is monotonic. It changes the score scale, but it does not meaningfully change the ranking order of examples.

Because AUROC and AUPRC depend on ranking, not absolute probability values, they stay unchanged.

## Risk-coverage curve

File:

- `phantom_4000_risk_coverage_curve.png`

What this plot means:

- x-axis: coverage, the fraction of examples the system still answers
- y-axis: selective risk, the unsupported rate among the answers it keeps

How to read it:

- moving left means abstaining on more examples
- if the score is useful, selective risk should fall as coverage falls

What happened here:

- the minimum coverage constraint was `0.8`
- the frozen validation threshold was `0.5132`
- on validation, coverage was `0.8040` and selective risk was `0.2107`
- on test, coverage was `0.8185` and selective risk was `0.2137`

Interpretation:

- Abstaining on about `18 to 20%` of the riskiest examples lowers the unsupported rate of the kept answers from about `31 to 33%` down to about `21%`.
- This is strong evidence that the calibrated risk score is useful for selective rejection.

## Accuracy-coverage curve

File:

- `phantom_4000_accuracy_coverage_curve.png`

What this plot means:

- x-axis: coverage
- y-axis: selective accuracy, meaning accuracy on only the examples the system keeps

How to read it:

- moving left means the system answers fewer examples
- if abstention helps, selective accuracy should rise as the riskiest examples are removed

What happened here:

- at full coverage, validation selective accuracy was `0.6744`
- at the selected validation operating point, validation selective accuracy rose to `0.7893`
- at full coverage, test selective accuracy was `0.6832`
- at the frozen-threshold test point, test selective accuracy rose to `0.7863`

Interpretation:

- Once the riskiest cases are removed, the remaining answered set becomes much more reliable.
- This is the positive side of the risk-coverage result.

## Selected operating point

Selection rule:

> Choose the validation risk threshold that maximizes selective accuracy subject to the minimum coverage constraint.

Chosen validation operating point:

- threshold: `0.5132`
- coverage: `0.8040`
- abstention rate: `0.1960`
- selective risk: `0.2107`
- selective accuracy: `0.7893`

Frozen-threshold test operating point:

- reported threshold on test curve: `0.4986`
- coverage: `0.8185`
- abstention rate: `0.1815`
- selective risk: `0.2137`
- selective accuracy: `0.7863`

The slight threshold mismatch between validation and test is due to the test curve being evaluated on the discrete set of test score values. The decision policy is still the frozen validation rule.

## Main conclusion

This result supports three points:

1. The tuned PHANTOM detector learns sensible risk signals.
2. Platt scaling is a reasonable final calibration choice.
   It improves validation calibration and improves several threshold-based decision metrics on test, even though test probability calibration is mixed.
3. The abstention rule works well.
   Abstaining on roughly the riskiest `20%` of cases raises the quality of the kept answers substantially while preserving about `80%` coverage.

In short:

> The calibrated PHANTOM risk score is useful for rejecting risky examples. By abstaining on about one fifth of cases, the system keeps a much cleaner set of answers.
