# Calibration and Abstention Findings

## What was done

After training the tuned logistic regression detector, the next step was to calibrate its risk scores and turn them into an abstention rule.

The workflow was:

1. Use the tuned PHANTOM detector.
2. Fit temperature scaling on the validation split.
3. Convert the detector output into calibrated risk scores.
4. Measure calibration with ECE and Brier score.
5. Choose an abstention threshold on validation.
6. Freeze that threshold.
7. Evaluate the frozen rule on test.

The generated plots explain these steps.

## Reliability diagram

Files:

- `phantom_4000_reliability_diagram_val.png`
- `phantom_4000_reliability_diagram_test.png`

What this plot means:

- the x-axis is the predicted hallucination risk
- the y-axis is the actual unsupported rate
- the diagonal line is perfect calibration

How to read it:

- points near the diagonal mean the predicted risk is trustworthy
- points below the diagonal mean the model predicts too much risk
- points above the diagonal mean the model predicts too little risk

What happened here:

- temperature scaling helped validation ECE slightly
- validation ECE changed from `0.0694` to `0.0670`
- validation Brier stayed almost the same, from `0.1514` to `0.1515`
- on test, calibration became slightly worse
- test ECE changed from `0.0598` to `0.0705`
- test Brier changed from `0.1695` to `0.1704`

Interpretation:

- the raw tuned detector was already fairly reasonable as a risk model
- temperature scaling did not dramatically improve calibration
- it gave only a small validation improvement and did not transfer cleanly to test

So this plot does not show a major calibration win. It shows that calibration was attempted correctly, gave a small in-domain gain, and was somewhat brittle out of domain even within the PHANTOM split.

## Risk-coverage curve

File:

- `phantom_4000_risk_coverage_curve.png`

What this plot means:

- the x-axis is coverage, which is the fraction of examples we still answer
- the y-axis is selective risk, which is the unsupported rate among only the answers we keep

How to read it:

- moving left means we abstain on more examples
- if the detector is useful, risk should go down as coverage goes down
- that means the model is correctly rejecting the riskier cases first

What happened here:

- the selected validation operating point used a frozen threshold of `0.3986`
- on validation, coverage was `0.8040`
- on validation, selective risk was `0.2107`
- on test, coverage was `0.8185`
- on test, selective risk was `0.2137`

Interpretation:

- after abstaining on about `18 to 20%` of the riskiest examples, the remaining answered examples are much safer
- the kept answers have only about `21%` unsupported rate
- this means the abstention rule is doing what it is supposed to do

This is one of the strongest plots because it directly shows that the risk score is useful for selective answering.

## Accuracy-coverage curve

File:

- `phantom_4000_accuracy_coverage_curve.png`

What this plot means:

- the x-axis is coverage
- the y-axis is selective accuracy, which is the accuracy on only the examples we keep

How to read it:

- moving left means we answer fewer examples
- if abstention helps, the accuracy of the kept examples should increase

What happened here:

- at the selected validation threshold, validation selective accuracy was `0.7893`
- at the frozen threshold on test, selective accuracy was `0.7863`

Interpretation:

- once the riskiest examples are removed, the accuracy of the remaining answers becomes noticeably higher
- this is the positive of the risk-coverage result

Together, the risk-coverage and accuracy-coverage plots show that abstention improves the quality of the answers that are still returned.

## Selected operating point

What this means:

- the operating point is the single abstention threshold chosen on validation
- after that, it is frozen and reused on test

Why this matters:

- this avoids tuning on the test set
- it makes the abstention evaluation fair
- it gives a concrete rule that can later be transferred to WikiQA

The frozen rule here is:

- if calibrated risk is greater than `0.3986`, abstain
- otherwise, answer

This is the rule that should be carried into the transfer setting when PHANTOM is the source regime.

## Does this prove the point

It supports part of the point clearly, and part of it only weakly.

What it supports clearly:

1. The detector scores are useful for ranking risky answers.
2. An abstention rule built from those scores improves the quality of kept answers.
3. The selected threshold transfers from validation to test in a stable way inside PHANTOM.

What it supports only weakly:

1. Temperature scaling did not give a strong calibration improvement.
2. Calibration quality did not improve on test.
3. So the main strength right now is selective abstention, not strong probability calibration.

So the claim is:

- the PHANTOM detector supports the risk-adjusted abstention point well
- the calibration story is mixed rather than strong

That is still a valid result. It means the model is useful for deciding when to abstain, even if its probability values are not perfectly calibrated.

## Simple conclusion

The new plots show that the detector can identify higher-risk answers and abstain on them effectively. This improves the safety of the answers that are kept. The abstention story is strong. The calibration story is more modest, because temperature scaling only helped slightly on validation and did not improve test calibration. So the current PHANTOM result supports the usefulness of the risk score for abstention more strongly than it supports perfectly calibrated risk probabilities.
