# Two-Dataset Results Guide

## What is done now

For the current standalone experiments, the main checklist is complete for both datasets.

- Feature generation is done.
- The feature table was built.
- Train, validation, and test splits were created.
- Z-score standardization was fit on the training split only.
- The same training-set mean and standard deviation were applied to train, validation, and test.
- Logistic regression detectors were trained.
- Baseline and ablation comparisons were run.
- AUROC, AUPRC, accuracy, precision, recall, and F1 were computed.
- Calibration methods were compared.
- A final calibrated risk score was selected.
- Abstention thresholds were chosen on validation.
- The thresholds were frozen.
- Risk-coverage and accuracy-coverage curves were produced.
- Frozen bundles were saved for later transfer experiments.

## How this lines up with the source papers

The code follows the main ideas from the source papers and the original project plan.

- Self-consistency follows the same basic idea as SelfCheckGPT: sample multiple answers and treat disagreement as a sign of hallucination risk.
  Source: Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models"https://aclanthology.org/2023.emnlp-main.557/
- Semantic entropy follows the same main idea as Farquhar et al.: cluster sampled answers by meaning, then measure uncertainty over semantic clusters instead of surface strings.
  Source: Farquhar et al., "Detecting hallucinations in large language models using semantic entropy"https://www.nature.com/articles/s41586-024-07421-0
- Calibration follows standard post-hoc calibration from Guo et al. and related work. The pipeline compares raw scores, temperature scaling, Platt scaling, and isotonic regression.
  Source: Guo et al., "On Calibration of Modern Neural Networks"https://proceedings.mlr.press/v70/guo17a.html
- Abstention follows the selective prediction idea: use a risk score, keep low-risk examples, and reject high-risk examples to improve the quality of answered cases.
  Source background: El-Yaniv and related selective classification work
  https://jmlr.org/beta/papers/v11/el-yaniv10a.html

Important interpretation point:

- This codebase is aligned with the main ideas from those papers.
- It is not a claim that every experiment is an exact reproduction of the original paper setups.
- The project goal here is a practical four-signal detector pipeline with calibration and abstention, not a paper-by-paper replication package.

## Main story across both datasets

The strongest overall result is not that one feature wins everywhere. The stronger result is that a reliability pipeline can be built end to end:

1. Generate multiple answers and evidence-aware features.
2. Train a detector on those features.
3. Calibrate the risk score.
4. Use the calibrated risk score for abstention.
5. Keep a frozen detector bundle for transfer later.

That is the core project contribution and it is now in place for both PHANTOM and WikiQA.

## PHANTOM interpretation

PHANTOM is the stronger clean success case.

- The full four-feature detector outperforms the single-feature baselines overall.
- On test, the full detector reaches about `AUROC = 0.7730` and `AUPRC = 0.6451`.
- The learned weights make intuitive sense:
  - self-consistency disagreement has the strongest positive weight
  - token uncertainty and semantic entropy are also positive
  - groundedness is negative, which means better evidence support lowers hallucination risk
- Platt scaling was the best practical final choice for PHANTOM.
- Abstention works well on PHANTOM:
  - full-coverage test accuracy is about `0.7591`
  - after abstaining on about `18.15%` of cases, selective accuracy rises to about `0.7863`

What to emphasize:

- PHANTOM is the clearest evidence that combining uncertainty and groundedness helps.
- PHANTOM also shows that calibrated risk is useful for abstention.
- This is the dataset where the detector story is the most coherent from features to final selective prediction.

## WikiQA interpretation

WikiQA is more mixed and should be presented that way.

- The full detector is still useful, but the margins are smaller than on PHANTOM.
- On test, the tuned detector reaches about `AUROC = 0.6927` and `AUPRC = 0.3434`.
- Token uncertainty is individually strong on WikiQA.
- Semantic entropy is also useful.
- Groundedness alone is weak.
- Self-consistency alone performs poorly on WikiQA in this setup.

The most important calibration result on WikiQA is:

- isotonic regression is better than Platt scaling on WikiQA for the final calibration metrics
- final WikiQA calibration therefore uses isotonic regression, not Platt scaling

Final WikiQA calibrated test values:

- `ECE = 0.0725`
- `Brier = 0.1656`

Abstention is weaker on WikiQA than on PHANTOM:

- full-coverage accuracy is about `0.7602`
- with the frozen abstention threshold, coverage stays very high at about `0.9643`
- selective accuracy is about `0.7566`

What to emphasize:

- WikiQA shows that calibration choice is dataset dependent.
- WikiQA is a good argument for comparing calibration methods instead of forcing one method everywhere.
- WikiQA is also a useful reminder that evidence regime matters. The same detector design does not behave identically across datasets.

What not to overclaim:

- Do not present WikiQA as a strong abstention gain story.
- Do not present WikiQA coefficients as cleanly interpretable in the same way as PHANTOM.
- Use WikiQA more as an evidence-regime contrast than as a perfect mirror of PHANTOM.

## Which images to use

## Best images for the poster

Use a small number of figures that support one clear story.

### 1. PHANTOM baseline comparison

Use:

- `results/plots/phantom_4000_baseline_metric_bars.png`

Why:

- This shows that the full detector is competitive with or better than the single-signal baselines.
- It supports the claim that combining signals is better than using only one signal.

### 2. PHANTOM learned coefficients

Use:

- `results/plots/phantom_4000_tuned_logreg_coefficients.png`

Why:

- This is the simplest way to show which signals the detector learned to trust.
- It helps explain the detector in one visual.

### 3. PHANTOM reliability or abstention figure

Use one of these, not both, unless you have room:

- `results/calibration/phantom_4000_reliability_diagram_test.png`
- `results/calibration/phantom_4000_risk_coverage_curve.png`
- `results/calibration/phantom_4000_accuracy_coverage_curve.png`

Recommendation:

- If the poster should stress calibration, use the reliability diagram.
- If the poster should stress safe deployment and selective answering, use the accuracy-coverage curve.

### 4. WikiQA calibration comparison

Use:

- `results/wikiqa/calibration_compare/wikiqa_1300_reliability_comparison_test.png`

Why:

- This is the clearest visual justification for using isotonic on WikiQA.
- It also shows that the second dataset behaves differently from PHANTOM.

## Best images for the written report

The report can include more detail.

### PHANTOM

- `results/plots/phantom_4000_baseline_metric_bars.png`
- `results/plots/phantom_4000_baseline_metric_heatmap.png`
- `results/plots/phantom_4000_tuned_logreg_coefficients.png`
- `results/plots/phantom_4000_roc_pr_curves.png`
- `results/calibration/phantom_4000_reliability_diagram_val.png`
- `results/calibration/phantom_4000_reliability_diagram_test.png`
- `results/calibration/phantom_4000_risk_coverage_curve.png`
- `results/calibration/phantom_4000_accuracy_coverage_curve.png`

### WikiQA

- `results/wikiqa/plots/wikiqa_1300_baseline_metric_bars.png`
- `results/wikiqa/plots/wikiqa_1300_baseline_metric_heatmap.png`
- `results/wikiqa/plots/wikiqa_1300_tuned_logreg_coefficients.png`
- `results/wikiqa/plots/wikiqa_1300_roc_pr_curves.png`
- `results/wikiqa/calibration_compare/wikiqa_1300_reliability_comparison_val.png`
- `results/wikiqa/calibration_compare/wikiqa_1300_reliability_comparison_test.png`
- `results/wikiqa/calibration/wikiqa_1300_reliability_diagram_test.png`
- `results/wikiqa/calibration/wikiqa_1300_risk_coverage_curve.png`
- `results/wikiqa/calibration/wikiqa_1300_accuracy_coverage_curve.png`

## Suggested poster emphasis

Given the poster format you shared, the poster should tell one simple story:

### Top message

Use a title and subtitle that focus on reliability:

- Combining uncertainty and groundedness improves hallucination detection.
- Calibration and abstention make the detector more usable in high-risk settings.

### Problem statement bullets

Good points to emphasize:

- LLMs can sound confident while being unsupported.
- No single signal captures hallucination risk well enough by itself.
- Evidence support and uncertainty should be combined.
- Risk scores need calibration before they are used operationally.

### Method bullets

Keep this simple:

- Sample multiple answers.
- Compute four features: token uncertainty, self-consistency disagreement, semantic entropy, groundedness.
- Train a logistic regression detector.
- Calibrate the detector score.
- Use the calibrated score for abstention.

### Results bullets

Recommended emphasis:

- PHANTOM: the combined detector is stronger than single-signal baselines.
- PHANTOM: Platt-scaled risk supports useful abstention, improving the quality of answered cases.
- WikiQA: the same pipeline still works, but calibration choice is different. Isotonic performs better than Platt.
- Across both datasets: the project completed the full standalone reliability pipeline and saved frozen bundles for transfer.

### Final takeaway

Best closing message:

- The project shows that hallucination detection is stronger when uncertainty and evidence support are combined, and that calibration plus abstention turns raw detector scores into a more reliable decision tool.

## Summary

A clean short interpretation is:

> Across both PHANTOM and WikiQA, we completed the full detector pipeline from feature generation to calibration and abstention. PHANTOM gives the clearest positive result: the combined detector outperforms simpler baselines, Platt scaling improves the usefulness of the risk score, and abstaining on the riskiest cases improves the quality of retained answers. WikiQA is more mixed, but still supports the broader project goal: the same pipeline remains usable across a different evidence regime, and the calibration comparison shows that the best post-hoc calibrator is dataset dependent, with isotonic regression outperforming Platt on WikiQA.

## Checklist status

For the current project state:

- Standalone PHANTOM pipeline: done
- Standalone WikiQA pipeline: done
- Calibration comparison: done on both datasets
- Final calibrated detector bundle: done on both datasets
- Abstention policy and operating point: done on both datasets

That means the major checklist is complete for the standalone experiments even though the datasets evolved during the project.
