# Calibration Method Comparison Findings

## What was compared

Three calibration methods were compared after training the tuned PHANTOM detector:

1. Temperature scaling
2. Platt scaling
3. Isotonic regression

Each method was fit on the validation split only and then evaluated on the test split.

The main calibration metrics reported were:

- ECE
- Brier score
- reliability diagrams

## Test-set results

### Raw detector

- ECE: `0.0598`
- Brier: `0.1695`

### Temperature scaling

- ECE: `0.0705`
- Brier: `0.1704`

### Platt scaling

- ECE: `0.0734`
- Brier: `0.1652`

### Isotonic regression

- ECE: `0.0937`
- Brier: `0.1715`

## Visual interpretation from the reliability diagrams

### Temperature scaling

- did not improve the test reliability diagram enough
- still showed noticeable zig-zag behavior in the middle and higher risk ranges
- did not give the best test ECE or the best test Brier score

### Platt scaling

- gave the smoothest and most stable calibration behavior across validation and test
- stayed closer to the diagonal than temperature scaling in most of the important middle-risk region
- gave the best test Brier score
- looked like the best practical compromise between smoothness and generalization

### Isotonic regression

- looked strong on validation
- looked less stable on test
- showed signs of overfitting, especially with sharper changes in the test reliability curve
- also had the worst test ECE and worst test Brier score among the calibrated methods

## Why Platt scaling is the main calibration choice

Platt scaling does not win on every metric, but it is still the best overall choice here.

Why:

1. It gives the best test Brier score.
2. It behaves more smoothly than temperature scaling in the reliability diagrams.
3. It generalizes better than isotonic regression.
4. It is still simple and easy to justify in the report.

The raw detector still has the best test ECE, so the correct claim is not that Platt scaling dominates every method. The correct claim is that Platt scaling gives the best overall practical calibration tradeoff for this project.

## Final decision

The main calibration pipeline should use Platt scaling instead of temperature scaling.

Temperature scaling should remain in the report as a calibration baseline.

Isotonic regression should also remain in the report as a comparison method, but not as the main calibrator.

## Simple conclusion

Platt scaling is the most reasonable main calibration method for the PHANTOM detector because it gives the best overall practical behavior: the cleanest reliability-diagram alignment across validation and test and the best test Brier score, while avoiding the instability seen with isotonic regression. Temperature scaling did not improve calibration enough to justify keeping it as the main method.
