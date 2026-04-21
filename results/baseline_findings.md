# Baseline Findings

## Main result

The full 4-feature detector is the best overall baseline result on PHANTOM.

Test metrics for the full model:

- AUROC: 0.773
- AUPRC: 0.645
- Accuracy: 0.759
- Precision: 0.610
- Recall: 0.667
- F1: 0.637

Simple meaning:

- it separates supported and unsupported answers better than any single feature alone
- it gives the best overall balance across ranking quality and classification quality

## Single-feature baselines

### Self-consistency only

This is the strongest single-feature baseline.

Test metrics:

- AUROC: 0.762
- AUPRC: 0.620
- Accuracy: 0.686
- Precision: 0.504
- Recall: 0.729
- F1: 0.596

Simple meaning:

- when sampled answers disagree more, hallucination risk tends to go up
- this feature alone is strong, but still not as good as combining all four features

### Token uncertainty only

This is the second strongest single-feature baseline.

Test metrics:

- AUROC: 0.745
- AUPRC: 0.640
- Accuracy: 0.716
- Precision: 0.548
- Recall: 0.594
- F1: 0.570

Simple meaning:

- if the model is less confident in its tokens, that often matches higher hallucination risk
- this feature works reasonably well on its own
- the full model still does better overall

### Semantic entropy only

Test metrics:

- AUROC: 0.701
- AUPRC: 0.491
- Accuracy: 0.700
- Precision: 0.520
- Recall: 0.677
- F1: 0.588

Simple meaning:

- meaning-level variation across sampled answers has useful signal
- but it is weaker than self-consistency disagreement and token uncertainty by itself

### Groundedness only

This is the weakest single-feature baseline.

Test metrics:

- AUROC: 0.531
- AUPRC: 0.326
- Accuracy: 0.541
- Precision: 0.333
- Recall: 0.448
- F1: 0.382

Simple meaning:

- groundedness by itself is not enough to detect unsupported answers well on this PHANTOM setup
- it seems more useful as one part of a combined detector than as a standalone score

## Manual weighted baseline

Test metrics:

- AUROC: 0.715
- AUPRC: 0.545
- Accuracy: 0.706
- Precision: 0.537
- Recall: 0.531
- F1: 0.534

Simple meaning:

- manually choosing weights is worse than letting logistic regression learn them from data
- this supports the learned detector over the hand-tuned score

## Random baseline

Test metrics:

- AUROC: 0.483
- AUPRC: 0.323
- Accuracy: 0.337
- Precision: 0.316
- Recall: 0.938
- F1: 0.472

Simple meaning:

- this is basically near chance
- it confirms the trained models are learning structure from the features

## Overall takeaway

The baseline comparison supports these conclusions:

1. Using all four features together works better than using only one feature.
2. Learning the weights from data works better than setting the weights by hand.
3. The features give complementary information.

In simple words, the best signal comes from self-consistency disagreement, token uncertainty is also strong, semantic entropy helps somewhat, and groundedness is weak alone but still useful in the combined model.

## One-paragraph version

On PHANTOM, the combined 4-feature detector gave the best overall results and beat every single-feature baseline, the manual weighted score, and the random baseline. Self-consistency disagreement was the strongest single feature, token uncertainty was next, semantic entropy had moderate value, and groundedness alone was weak. This suggests that hallucination risk is captured best when uncertainty and groundedness signals are combined and the model learns how much to trust each one.
