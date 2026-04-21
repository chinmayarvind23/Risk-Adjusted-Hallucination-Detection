# Full Results Summary

This file collects the main findings from the whole repo so far.

It combines:

- the original project plan in [docs/initial_plan.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/docs/initial_plan.md)
- the PHANTOM detector findings in [results/findings_phantom.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/findings_phantom.md)
- the PHANTOM baseline findings in [results/baseline_findings.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/baseline_findings.md)
- the PHANTOM calibration findings in [results/calibration/calibration_findings.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/calibration_findings.md)
- the calibration method comparison findings in [results/calibration_compare/calibration_method_findings.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration_compare/calibration_method_findings.md)
- the two-dataset guide in [results/two_dataset_results_guide.md](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/two_dataset_results_guide.md)

The goal is simple:

- explain what was built
- explain what worked
- explain what did not transfer well
- explain which figures matter most

## 1. Project goal

The project goal was to build a **risk-adjusted hallucination detector**.

In simple words, the system tries to answer this question:

> Is the model answer supported by the evidence we gave it, or is it risky enough that we should not trust it?

The full pipeline does four things:

1. Generate answers and compute features.
2. Train a detector to predict unsupported answers.
3. Calibrate the detector score so it behaves more like a risk probability.
4. Abstain on high-risk examples instead of answering them.

The original plan used PHANTOM and a retrieval-grounded second regime. In the final repo, the second regime is **WikiQA**. The exact second dataset changed, but the main checklist stayed the same:

- one source dataset
- one target dataset
- same feature pipeline
- same detector family
- calibration
- abstention
- transfer in both directions

So the core project idea was kept intact.

## 2. Paper ideas used in the implementation

The code is based on the main ideas from these papers:

- Self-consistency and sampled-answer disagreement: Manakul et al., *SelfCheckGPT*https://aclanthology.org/2023.emnlp-main.557/
- Semantic clustering and entropy over meanings: Farquhar et al., *Detecting hallucinations in large language models using semantic entropy*https://www.nature.com/articles/s41586-024-07421-0
- Calibration: Guo et al., *On Calibration of Modern Neural Networks*https://proceedings.mlr.press/v70/guo17a.html
- Selective prediction and risk-coverage tradeoff: El-Yaniv and related work
  https://jmlr.org/beta/papers/v11/el-yaniv10a.html

Important note:

- the code follows the main method ideas from these papers
- it is not claiming exact paper reproduction
- it is a practical project implementation of those ideas

## 3. What was implemented

The final feature set has four parts:

1. **Token uncertainty**
   - higher mean token negative log-likelihood means the model was less confident in the tokens it generated
2. **Self-consistency disagreement**
   - sampled answers that disagree more suggest higher hallucination risk
3. **Semantic entropy**
   - if sampled answers spread across different meanings, uncertainty is higher
4. **Groundedness or evidence consistency**
   - if answer sentences are better supported by the evidence, hallucination risk should go down

These features are then used in a logistic regression detector.

After that:

- calibration methods are compared
- a final calibration method is chosen
- an abstention threshold is selected on validation
- a frozen detector bundle is saved

## 4. What is complete in the repo

The following parts are complete:

- PHANTOM standalone pipeline
- WikiQA standalone pipeline
- transfer experiment 1: PHANTOM to WikiQA
- transfer experiment 2: WikiQA to PHANTOM
- baseline comparisons
- calibration comparisons
- abstention curves
- frozen bundles for both datasets

So the full planned flow is now present in the repo.

## 5. PHANTOM standalone findings

PHANTOM is the cleanest success case in the project.

### 5.1 Detector quality

From the tuned detector report:

- AUROC: `0.7730`
- AUPRC: `0.6451`
- accuracy: `0.7591`
- precision: `0.6095`
- recall: `0.6667`
- F1: `0.6368`

This means the detector separates supported and unsupported answers fairly well on PHANTOM.

### 5.2 What the learned weights mean

The PHANTOM tuned coefficients are:

- `mean_token_nll = +0.3014`
- `self_consistency_disagreement = +0.7280`
- `semantic_entropy = +0.2007`
- `groundedness_score = -0.3692`

Simple reading:

- more disagreement means more risk
- more token uncertainty means more risk
- more semantic entropy means more risk
- more groundedness means less risk

This is exactly the pattern we hoped for.

### 5.3 Baseline comparison

The strongest single feature on PHANTOM is **self-consistency disagreement**.

But the full detector is still best overall.

That supports the main claim:

> combining uncertainty and groundedness works better than using one signal alone

Best PHANTOM baseline figure:

- [phantom_4000_baseline_metric_bars.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/plots/phantom_4000_baseline_metric_bars.png)

What to notice in the figure:

- the full four-feature detector is the strongest overall bar pattern
- self-consistency only is strong, but still lower
- groundedness alone is weak
- the combined model gets the benefit of complementary signals

### 5.4 Calibration on PHANTOM

PHANTOM uses **Platt scaling** as the final calibration method.

Why:

- it gave the best practical tradeoff in the PHANTOM comparison
- it was more stable than isotonic on PHANTOM test data

Final calibrated PHANTOM test metrics:

- ECE: `0.0734`
- Brier: `0.1652`

Best PHANTOM calibration figures:

- [phantom_4000_reliability_diagram_test.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_reliability_diagram_test.png)
- [phantom_4000_risk_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_risk_coverage_curve.png)
- [phantom_4000_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_accuracy_coverage_curve.png)

What to notice:

- the reliability diagram shows the score is at least usable as a risk signal
- the risk-coverage curve slopes in the right direction, which means dropping risky examples lowers error
- the accuracy-coverage curve rises as coverage falls, which means abstention is helping

### 5.5 Abstention on PHANTOM

At full coverage, PHANTOM test selective accuracy is about `0.6832`.

At the frozen operating point:

- coverage: `0.8185`
- abstention rate: `0.1815`
- selective accuracy: `0.7863`

Simple interpretation:

- the system refuses about 18 percent of the riskiest cases
- the quality of the remaining answers goes up a lot

This is one of the strongest results in the repo.

## 6. WikiQA standalone findings

WikiQA is a weaker but still useful result.

### 6.1 Detector quality

From the tuned WikiQA detector report:

- AUROC: `0.6927`
- AUPRC: `0.3434`
- accuracy: `0.6837`
- precision: `0.4096`
- recall: `0.7234`
- F1: `0.5231`

This is clearly weaker than PHANTOM, but it is still above a trivial detector.

### 6.2 What the weights mean

The WikiQA tuned coefficients are:

- `mean_token_nll = -0.3388`
- `self_consistency_disagreement = +0.1640`
- `semantic_entropy = -0.0561`
- `groundedness_score = -0.2647`

This is less clean than PHANTOM.

What this suggests:

- groundedness still helps in the expected direction
- self-consistency still adds some positive risk signal
- the feature interactions are more unstable on WikiQA

So WikiQA should be presented as a harder evidence regime, not as a second clean win.

### 6.3 Baseline comparison

On WikiQA:

- token uncertainty is strong by itself
- semantic entropy is also useful
- groundedness alone is weak
- self-consistency alone is poor in this setup

Best WikiQA standalone figures:

- [wikiqa_1300_baseline_metric_bars.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/plots/wikiqa_1300_baseline_metric_bars.png)
- [wikiqa_1300_tuned_logreg_coefficients.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/plots/wikiqa_1300_tuned_logreg_coefficients.png)

What to notice:

- the gap between the full detector and the single-feature baselines is smaller than on PHANTOM
- the learned weights are less intuitive than on PHANTOM

That is a useful result by itself. It shows that the evidence regime matters.

### 6.4 Calibration on WikiQA

WikiQA uses **isotonic regression** as the final calibration method.

Why:

- on WikiQA, isotonic gave the best calibration metrics among the tested methods

Final calibrated WikiQA test metrics:

- ECE: `0.0725`
- Brier: `0.1656`

Best WikiQA calibration comparison figures:

- [wikiqa_1300_reliability_comparison_test.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration_compare/wikiqa_1300_reliability_comparison_test.png)
- [wikiqa_1300_reliability_diagram_test.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration/wikiqa_1300_reliability_diagram_test.png)

What to notice:

- WikiQA does not want the same calibrator as PHANTOM
- this supports the claim that calibration choice is dataset dependent

### 6.5 Abstention on WikiQA

At the frozen WikiQA operating point:

- coverage: `0.9643`
- abstention rate: `0.0357`
- selective accuracy: `0.7566`

Simple interpretation:

- the abstention gain on WikiQA is much smaller than on PHANTOM
- the system barely abstains, so there is less room for selective improvement

This should be described. WikiQA is not the main abstention success story.

## 7. Transfer experiment 1: PHANTOM to WikiQA

Transfer experiment 1 is complete.

Files:

- [phantom_to_wikiqa_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_transfer_report.json)
- [phantom_to_wikiqa_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_reliability_diagram.png)
- [phantom_to_wikiqa_risk_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_risk_coverage_curve.png)
- [phantom_to_wikiqa_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_accuracy_coverage_curve.png)

Calibrated transfer metrics:

- AUROC: `0.3996`
- AUPRC: `0.2020`
- accuracy: `0.7554`
- precision: `0.2736`
- recall: `0.1074`
- F1: `0.1543`
- ECE: `0.1561`
- Brier: `0.1985`

At the frozen PHANTOM threshold on WikiQA:

- coverage: `0.9238`
- abstention rate: `0.0762`
- selective accuracy: `0.7960`

Simple interpretation:

- the PHANTOM detector does not transfer well to WikiQA
- ranking quality drops sharply
- calibration gets worse
- the model still answers most cases, but the risk score is no longer a good general detector

What to notice in the transfer figures:

- the reliability diagram is clearly worse than the in-domain PHANTOM and in-domain WikiQA diagrams
- the selective curves do not show the same clean safety gain pattern as the source-domain results

This is a real result, not a failure of the experiment. It means the transfer question was tested and the answer was mostly negative.

## 8. Transfer experiment 2: WikiQA to PHANTOM

Transfer experiment 2 is also complete.

Files:

- [wikiqa_to_phantom_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_transfer_report.json)
- [wikiqa_to_phantom_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_reliability_diagram.png)
- [wikiqa_to_phantom_risk_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_risk_coverage_curve.png)
- [wikiqa_to_phantom_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_accuracy_coverage_curve.png)

Calibrated transfer metrics:

- AUROC: `0.4294`
- AUPRC: `0.2943`
- accuracy: `0.7049`
- precision: `0.6000`
- recall: `0.0399`
- F1: `0.0748`
- ECE: `0.1659`
- Brier: `0.2495`

At the frozen WikiQA threshold on PHANTOM:

- coverage: `0.8510`
- abstention rate: `0.1490`
- selective accuracy: `0.7081`

Simple interpretation:

- WikiQA to PHANTOM also transfers poorly
- ranking quality is low
- recall is extremely low
- calibration is poor

This means the poor transfer is not just one bad direction. Both directions are weak.

## 9. Main cross-dataset conclusion

The transfer story is clear:

- the standalone pipelines work
- the cross-dataset transfer does not work well

That is one of the most important final conclusions in the repo.

In simple words:

> The detector can learn useful risk signals inside one evidence regime, but those signals do not carry over well to a different dataset without major loss.

This is a strong result. It suggests that:

- hallucination patterns depend on the evidence regime
- calibration is not stable under shift
- thresholds chosen on one dataset do not automatically stay useful on another

## 10. Deeper diagnostic conclusions

The basic transfer result is already clear from the transfer reports:

- [phantom_to_wikiqa_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_transfer_report.json)
- [wikiqa_to_phantom_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_transfer_report.json)

But to make the project look more like serious research work, it helps to answer **why** transfer fails, not just show that it fails.

The repo now includes a deeper diagnostic pass here:

- [transfer_diagnostics.py](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/analysis/transfer_diagnostics.py)
- [transfer_diagnostics.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/transfer_diagnostics.json)

This section summarizes those diagnostics.

### 10.1 Why transfer fails

The strongest answer is:

> transfer fails because several feature meanings change across datasets, the learned detector weights do not match across regimes, and source-side calibration is not portable.

This is not just a threshold problem.

The evidence for that comes from four places:

- the raw feature distribution comparison in [transfer_diagnostics.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/transfer_diagnostics.json)
- the standalone tuned detector reports
  - [phantom_4000_logreg_tuned_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_4000_logreg_tuned_report.json)
  - [wikiqa_1300_logreg_tuned_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/wikiqa_1300_logreg_tuned_report.json)
- the frozen bundles
  - [phantom_4000_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_frozen_bundle.json)
  - [wikiqa_1300_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration/wikiqa_1300_frozen_bundle.json)
- the transfer reports listed above

### 10.2 Which features shift most across datasets

From [transfer_diagnostics.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/transfer_diagnostics.json), the overall raw feature means already show a regime shift:

- PHANTOM overall means:
  - `mean_token_nll = 0.0619`
  - `self_consistency_disagreement = 0.0793`
  - `semantic_entropy = 0.2647`
  - `groundedness_score = 0.6535`
- WikiQA overall means:
  - `mean_token_nll = 0.0443`
  - `self_consistency_disagreement = 0.0349`
  - `semantic_entropy = 0.2304`
  - `groundedness_score = 0.8273`

Simple reading:

- PHANTOM answers look more uncertain on the three uncertainty features
- WikiQA answers look more grounded on average

That matters, but the more important result is the **class-conditional direction**.

Unsupported minus supported on PHANTOM:

- `mean_token_nll = +0.0868`
- `self_consistency_disagreement = +0.1335`
- `semantic_entropy = +0.2661`
- `groundedness_score = -0.1118`

Unsupported minus supported on WikiQA:

- `mean_token_nll = -0.0187`
- `self_consistency_disagreement = -0.0041`
- `semantic_entropy = -0.1018`
- `groundedness_score = -0.0714`

This is one of the most important findings in the repo.

On PHANTOM:

- unsupported answers are more uncertain
- unsupported answers are less grounded

On WikiQA:

- unsupported answers are still less grounded
- but they are **not** more uncertain on average
- for three uncertainty features, the direction flips

So the strongest feature-level conclusion is:

> the uncertainty signals do not just weaken across datasets. Their relationship to the label changes direction. Groundedness is the only feature that keeps the same sign across both datasets.

This is why transfer is hard.

### 10.3 Do the learned coefficients mismatch across datasets

Yes, clearly.

From [phantom_4000_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_frozen_bundle.json), PHANTOM learned:

- `mean_token_nll = +0.3014`
- `self_consistency_disagreement = +0.7280`
- `semantic_entropy = +0.2007`
- `groundedness_score = -0.3692`

From [wikiqa_1300_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration/wikiqa_1300_frozen_bundle.json), WikiQA learned:

- `mean_token_nll = -0.3388`
- `self_consistency_disagreement = +0.1640`
- `semantic_entropy = -0.0561`
- `groundedness_score = -0.2647`

This means:

- token uncertainty flips sign
- semantic entropy flips sign
- self-consistency stays positive but becomes much weaker
- groundedness stays negative in both datasets

So the coefficient mismatch is real and strong. The transferred detector is not just using the wrong threshold. It is using the wrong **direction** for multiple signals.

### 10.4 Does calibration fail more than ranking, or does ranking fail first

Ranking already fails badly under transfer, before thresholding.

From [phantom_to_wikiqa_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_transfer_report.json):

- raw AUROC: `0.3996`
- raw AUPRC: `0.2020`

From [wikiqa_to_phantom_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_transfer_report.json):

- raw AUROC: `0.4471`
- raw AUPRC: `0.3148`

Those AUROC values are below `0.45` in both directions. So the ranking itself is already poor.

Calibration also fails under transfer.

PHANTOM to WikiQA:

- raw ECE: `0.1115`
- calibrated ECE: `0.1561`
- raw Brier: `0.1838`
- calibrated Brier: `0.1985`

WikiQA to PHANTOM:

- raw ECE: `0.1094`
- calibrated ECE: `0.1659`
- raw Brier: `0.2304`
- calibrated Brier: `0.2495`

So the correct conclusion is:

> ranking fails first, and calibration fails on top of that. Source-side calibration can even make target-domain calibration metrics worse.

This is stronger than simply saying transfer performance dropped.

### 10.5 Ranking quality versus decision quality

This distinction is important.

In-domain, PHANTOM has useful ranking quality:

- standalone AUROC: `0.7730`
- standalone AUPRC: `0.6451`

That same PHANTOM detector, when moved to WikiQA, drops to:

- transfer raw AUROC: `0.3996`
- transfer raw AUPRC: `0.2020`

The same pattern appears in the other direction:

- WikiQA standalone AUROC: `0.6927`
- WikiQA standalone AUPRC: `0.3434`
- WikiQA to PHANTOM raw AUROC: `0.4471`
- WikiQA to PHANTOM raw AUPRC: `0.3148`

Decision quality through thresholding also becomes unstable.

PHANTOM in-domain abstention gain from [transfer_diagnostics.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/transfer_diagnostics.json):

- full-coverage accuracy: `0.6832`
- selective accuracy at frozen point: `0.7863`
- gain: `+0.1031`

WikiQA in-domain:

- full-coverage accuracy: `0.7602`
- selective accuracy at frozen point: `0.7566`
- gain: `-0.0036`

Transfer 1:

- full target accuracy: `0.7554`
- selective accuracy: `0.7960`
- gain: `+0.0406`

Transfer 2:

- full target accuracy: `0.7049`
- selective accuracy: `0.7081`
- gain: `+0.0032`

So the cleaner statement is:

> in-domain, the detector can rank risk usefully, especially on PHANTOM. Under shift, ranking quality collapses, and the frozen abstention rule becomes much less useful.

### 10.6 Breakdown of failure cases

The repo does not yet include hand-labeled qualitative case studies, but the diagnostics do support one strong failure-case pattern:

> many unsupported target examples are still kept below the frozen source threshold.

From [transfer_diagnostics.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/transfer_diagnostics.json):

PHANTOM to WikiQA:

- unsupported target examples: `270`
- unsupported examples kept below the frozen threshold: `244`
- kept share within unsupported target examples: `0.9037`

WikiQA to PHANTOM:

- unsupported target examples: `602`
- unsupported examples kept below the frozen threshold: `500`
- kept share within unsupported target examples: `0.8306`

This is a strong version of the "confident wrong" problem in transfer:

- the detector is not rejecting most target unsupported cases
- many risky target cases still look safe to the source-domain model

That is consistent with the feature-direction flips above. If the source detector expects uncertainty to go up on unsupported cases, but the target dataset does not behave that way, many bad target cases will be scored too low.

### 10.7 Is the main problem evidence quality, label definition, class balance, or feature instability

The data in the repo supports **feature instability across evidence regimes** as the main problem.

Why this is the strongest supported answer:

- the label rate changes, but not enough to explain the full collapse
  - PHANTOM unsupported rate: `0.2991`
  - WikiQA unsupported rate: `0.2077`
- the coefficient patterns are not stable across datasets
- three of the four label-conditioned feature directions flip across datasets
- both transfer directions fail, which argues against a one-off issue in just one dataset

Evidence quality and dataset design likely contribute to that instability.

This is also consistent with how the groundedness feature is computed in [evidence_consistency.py](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/features/evidence_consistency.py), and how the uncertainty features are computed in:

- [token_entropy.py](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/features/token_entropy.py)
- [self_consistency.py](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/features/self_consistency.py)
- [semantic_entropy.py](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/features/semantic_entropy.py)

Those feature definitions are stable in code, but the target dataset makes them behave differently.

So the best-supported conclusion is:

> the pipeline does not fail because the features disappear. It fails because the meaning of several signals changes across datasets.

### 10.8 What part of the pipeline is robust

The most robust part is **groundedness**.

Evidence:

- its label direction is stable across both datasets
- its learned coefficient sign is stable across both standalone detectors

The least robust parts are:

- token uncertainty
- semantic entropy

Evidence:

- both flip sign in the label-conditioned feature comparison
- both also flip sign in the learned coefficients

Self-consistency sits in the middle:

- it is strong on PHANTOM
- it stays positive in the learned models
- but it is much weaker on WikiQA and its label-conditioned direction flips there

At the pipeline level:

- feature extraction is robust in the sense that it runs consistently across datasets
- detector fitting is not robust across regimes because the learned weight pattern changes
- calibration is not robust under transfer
- abstention thresholds are not portable in a strong way

This gives a sharper final research-style claim:

> groundedness is the most portable feature family in this project, while uncertainty-based features are more regime-dependent. The main bottleneck in transfer is not feature extraction itself, but the instability of learned decision rules and calibration under dataset shift.

### 10.9 How to verify or extend this analysis

The diagnostic script can be rerun with:

```powershell
python code\analysis\transfer_diagnostics.py `
  --phantom-raw data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv `
  --wikiqa-raw data\wiki_qa\train\wikiqa_1300_feature_table.csv `
  --phantom-tuned-report results\phantom_4000_logreg_tuned_report.json `
  --wikiqa-tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --phantom-calibration-report results\calibration\phantom_4000_calibration_abstention_report.json `
  --wikiqa-calibration-report results\wikiqa\calibration\wikiqa_1300_calibration_abstention_report.json `
  --phantom-bundle results\calibration\phantom_4000_frozen_bundle.json `
  --wikiqa-bundle results\wikiqa\calibration\wikiqa_1300_frozen_bundle.json `
  --phantom-to-wikiqa-report results\phantom_to_wikiqa_transfer\phantom_to_wikiqa_transfer_report.json `
  --wikiqa-to-phantom-report results\wikiqa_to_phantom_transfer\wikiqa_to_phantom_transfer_report.json `
  --wikiqa-transfer-standardized data\wiki_qa\train\wikiqa_1300_feature_table_standardized.csv `
  --phantom-transfer-standardized data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv `
  --output-json results\transfer_diagnostics.json
```

This output is the main citation file for the deeper transfer conclusions.

## 11. What the figures show overall

### Best PHANTOM figures

- [phantom_4000_baseline_metric_bars.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/plots/phantom_4000_baseline_metric_bars.png)
- [phantom_4000_tuned_logreg_coefficients.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/plots/phantom_4000_tuned_logreg_coefficients.png)
- [phantom_4000_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_accuracy_coverage_curve.png)

Pattern:

- the combined detector is best
- the learned weights make sense
- abstention clearly helps

### Best WikiQA figures

- [wikiqa_1300_baseline_metric_bars.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/plots/wikiqa_1300_baseline_metric_bars.png)
- [wikiqa_1300_reliability_comparison_test.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration_compare/wikiqa_1300_reliability_comparison_test.png)

Pattern:

- the detector still has signal
- the feature story is less clean than PHANTOM
- isotonic is the better final calibrator here

### Best transfer figures

- [phantom_to_wikiqa_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_reliability_diagram.png)
- [wikiqa_to_phantom_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_reliability_diagram.png)

Pattern:

- both transfer directions are much worse than the in-domain runs
- calibration drifts badly
- this supports the conclusion that cross-regime transfer is difficult

## 12. What to emphasize in the report

The report should emphasize three points:

1. **The main pipeline works in-domain**

   - feature generation
   - detector training
   - calibration
   - abstention
2. **PHANTOM is the strongest positive result**

   - clean detector improvement
   - sensible coefficients
   - useful abstention gains
3. **Transfer is the main limitation**

   - both transfer directions are weak
   - this shows the detector does not generalize well across datasets without adaptation

That is a strong and believable final story.

## 13. What to emphasize in the poster

The poster should be simpler than the report.

Recommended poster flow:

### Problem

- LLMs can give fluent but unsupported answers.
- A single signal is not enough to detect this well.

### Method

- sample answers
- compute four signals
- train a simple detector
- calibrate the risk
- abstain on risky answers

### Result

- PHANTOM: combined detector beats simpler baselines
- PHANTOM: abstention improves the quality of kept answers
- WikiQA: the same pipeline still works, but less cleanly
- transfer in both directions is weak

### Main takeaway

> Combining uncertainty and groundedness helps inside one dataset, but cross-dataset transfer remains hard.

## 14. Limitations

The repo supports these limitations:

- the method is grounded in paper ideas, but it is not a direct reproduction package
- WikiQA behaves differently from PHANTOM
- calibration method choice changes by dataset
- transfer is poor in both directions
- the feature set is useful, but not universal

These limitations do not weaken the project. They make the conclusions more credible.

## 15. Final bottom line

The full repo now supports this final conclusion:

> A four-feature hallucination detector built from token uncertainty, self-consistency disagreement, semantic entropy, and groundedness can work reasonably well within a dataset, especially on PHANTOM. Calibration and abstention make the detector more useful as a reliability tool. However, when the detector is frozen and moved across datasets, performance drops sharply in both directions, which shows that evidence regime and dataset shift matter a lot.

That is the clearest summary of the repo as it stands.
