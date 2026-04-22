# Risk-Adjusted Hallucination Detection

This repository studies hallucination detection for grounded question answering using four signals:

- token uncertainty
- self-consistency disagreement
- semantic entropy
- groundedness or evidence consistency

The full pipeline does five things:

1. generate answers and feature JSON
2. build a flat feature table
3. train a logistic regression detector
4. calibrate the detector risk score and choose an abstention threshold
5. test both in-domain and cross-dataset transfer

The final repo uses two datasets:

- **PHANTOM**
- **WikiQA**

Shared data folder:

- https://drive.google.com/drive/folders/1aHTuwsl0TuDfcwWspzaYjyUueDXrDSw5?usp=sharing

The transfer experiments are:

- **Transfer 1:** train on PHANTOM, calibrate on PHANTOM validation, freeze detector plus calibration plus threshold, test on WikiQA
- **Transfer 2:** train on WikiQA, calibrate on WikiQA validation, freeze detector plus calibration plus threshold, test on PHANTOM

This README is written for two use cases:

1. reproduce the final results from the prepared feature tables already in the repo
2. rerun the full workflow from raw feature JSON

## Repo layout

- [code/data_gen](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/data_gen)
  - data preparation, feature-table building, splitting, standardization
- [code/features](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/features)
  - feature computation
- [code/llm_generations](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/llm_generations)
  - answer generation plus feature extraction
- [code/detector](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/detector)
  - baselines, detector training, calibration, abstention, plots, transfer evaluation
- [code/analysis](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/code/analysis)
  - transfer diagnostics
- [data](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data)
  - feature JSON, flat CSVs, splits, standardized CSVs
- [results](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results)
  - reports, frozen bundles, plots, markdown findings

## Environment setup

This repo was run from Windows PowerShell with a local `venv`.

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
pip install -r code\requirements.txt
```

### 3. Optional quick syntax check

```powershell
python -m compileall code
```

## Quickstart

If you already have the prepared PHANTOM and WikiQA feature tables in `data/`, this is the shortest path to regenerate the main standalone and transfer results.

### PHANTOM standalone

```powershell
python code\data_gen\split_feature_table.py `
  --input data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv `
  --output-dir data\full_run\splits `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --prefix phantom_4000
```

```powershell
python code\data_gen\standardize_feature_splits.py `
  --train data\full_run\splits\phantom_4000_train.csv `
  --val data\full_run\splits\phantom_4000_val.csv `
  --test data\full_run\splits\phantom_4000_test.csv `
  --output-dir data\full_run\splits `
  --prefix phantom_4000
```

```powershell
python code\detector\run_feature_baselines.py `
  --train data\full_run\splits\phantom_4000_train_standardized.csv `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results `
  --prefix phantom_4000
```

```powershell
python code\detector\train_logreg_detector.py `
  --train data\full_run\splits\phantom_4000_train_standardized.csv `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results `
  --prefix phantom_4000 `
  --tune
```

```powershell
python code\detector\compare_calibration_methods.py `
  --tuned-report results\phantom_4000_logreg_tuned_report.json `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results\calibration_compare `
  --prefix phantom_4000
```

```powershell
python code\detector\calibrate_and_abstain.py `
  --tuned-report results\phantom_4000_logreg_tuned_report.json `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --standardization-stats data\full_run\splits\phantom_4000_standardization_stats.json `
  --output-dir results\calibration `
  --prefix phantom_4000 `
  --source-name phantom `
  --calibration-method platt
```

### WikiQA standalone

```powershell
python code\data_gen\split_feature_table.py `
  --input data\wiki_qa\train\wikiqa_1300_feature_table.csv `
  --output-dir data\wiki_qa\splits `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --prefix wikiqa_1300
```

```powershell
python code\data_gen\standardize_feature_splits.py `
  --train data\wiki_qa\splits\wikiqa_1300_train.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test.csv `
  --output-dir data\wiki_qa\splits `
  --prefix wikiqa_1300
```

```powershell
python code\detector\run_feature_baselines.py `
  --train data\wiki_qa\splits\wikiqa_1300_train_standardized.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa `
  --prefix wikiqa_1300
```

```powershell
python code\detector\train_logreg_detector.py `
  --train data\wiki_qa\splits\wikiqa_1300_train_standardized.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa `
  --prefix wikiqa_1300 `
  --tune
```

```powershell
python code\detector\compare_calibration_methods.py `
  --tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa\calibration_compare `
  --prefix wikiqa_1300
```

```powershell
python code\detector\calibrate_and_abstain.py `
  --tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --standardization-stats data\wiki_qa\splits\wikiqa_1300_standardization_stats.json `
  --output-dir results\wikiqa\calibration `
  --prefix wikiqa_1300 `
  --source-name wikiqa `
  --calibration-method isotonic
```

### Transfer 1: PHANTOM to WikiQA

```powershell
python code\data_gen\apply_standardization_stats.py `
  --stats-json results\calibration\phantom_4000_frozen_bundle.json `
  --inputs data\wiki_qa\train\wikiqa_1300_feature_table.csv `
  --output-dir data\wiki_qa\train
```

```powershell
python code\detector\evaluate_frozen_bundle.py `
  --bundle results\calibration\phantom_4000_frozen_bundle.json `
  --test data\wiki_qa\train\wikiqa_1300_feature_table_standardized.csv `
  --output-dir results\phantom_to_wikiqa_transfer `
  --prefix phantom_to_wikiqa `
  --source-name phantom `
  --target-name wikiqa
```

### Transfer 2: WikiQA to PHANTOM

```powershell
python code\data_gen\apply_standardization_stats.py `
  --stats-json results\wikiqa\calibration\wikiqa_1300_frozen_bundle.json `
  --inputs data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv `
  --output-dir data\full_run
```

```powershell
python code\detector\evaluate_frozen_bundle.py `
  --bundle results\wikiqa\calibration\wikiqa_1300_frozen_bundle.json `
  --test data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv `
  --output-dir results\wikiqa_to_phantom_transfer `
  --prefix wikiqa_to_phantom `
  --source-name wikiqa `
  --target-name phantom
```

### Deeper diagnostics

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

## End-to-end pipeline

The core detector workflow is:

1. Build feature table from JSON.
2. Split into train, validation, and test.
3. Compute training-set mean and standard deviation for each feature column.
4. Standardize train, validation, and test using those same training statistics.
5. Train the logistic regression detector.
6. Run baseline comparisons.
7. Compare calibration methods.
8. Choose the final calibrator.
9. Use the calibrated risk score for abstention.
10. Choose the abstention threshold on validation.
11. Freeze detector, calibration, and threshold.
12. Evaluate in-domain and transfer performance.

Important standardization rule:

- z-score standardize **within each feature column**
- do **not** standardize across rows
- do **not** use validation or test statistics to fit the scaler
- do this only for detector training and evaluation, not during answer generation

## Feature generation from raw data

If you need to prepare local PHANTOM or WikiQA subsets first:

```powershell
python code\data_gen\main.py --dataset phantom --num-rows 4000
python code\data_gen\main.py --dataset wikiqa --split train --num-rows 1300 --retrieve
```

If you need to generate answers and feature JSON directly:

```powershell
python code\llm_generations\main.py `
  --dataset phantom `
  --data-file <path_to_input_jsonl> `
  --output-file <path_to_output_json> `
  --model qwen3:8b `
  --judge-model facebook/bart-large-mnli `
  --num-rows 4000 `
  --k 5
```

```powershell
python code\llm_generations\main.py `
  --dataset wikiqa `
  --data-file <path_to_input_jsonl> `
  --output-file <path_to_output_json> `
  --model qwen3:8b `
  --judge-model facebook/bart-large-mnli `
  --num-rows 1300 `
  --k 5
```

If feature JSON was produced in multiple chunks, merge them:

```powershell
python code\data_gen\merge_feature_jsons.py `
  --inputs <chunk1.json> <chunk2.json> <chunk3.json> `
  --output <merged_features.json>
```

## Build feature table from JSON

This converts merged generation-feature JSON into a flat CSV used by the detector scripts.

### PHANTOM

```powershell
python code\data_gen\build_feature_table.py `
  --input data\full_run\qwen3_8b_k5_phantom_4000_features_deduped.json `
  --output-csv data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv
```

### WikiQA

```powershell
python code\data_gen\build_feature_table.py `
  --input data\wiki_qa\train\features\qwen3_8b_k5_wikiqa_1300_features.json `
  --output-csv data\wiki_qa\train\wikiqa_1300_feature_table.csv
```

## Train and evaluate the PHANTOM standalone pipeline

### 1. Split the PHANTOM feature table

```powershell
python code\data_gen\split_feature_table.py `
  --input data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv `
  --output-dir data\full_run\splits `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --prefix phantom_4000
```

### 2. Fit training-set z-score stats and transform train, validation, and test

```powershell
python code\data_gen\standardize_feature_splits.py `
  --train data\full_run\splits\phantom_4000_train.csv `
  --val data\full_run\splits\phantom_4000_val.csv `
  --test data\full_run\splits\phantom_4000_test.csv `
  --output-dir data\full_run\splits `
  --prefix phantom_4000
```

Outputs include:

- [phantom_4000_train_standardized.csv](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/full_run/splits/phantom_4000_train_standardized.csv)
- [phantom_4000_val_standardized.csv](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/full_run/splits/phantom_4000_val_standardized.csv)
- [phantom_4000_test_standardized.csv](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/full_run/splits/phantom_4000_test_standardized.csv)
- [phantom_4000_standardization_stats.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/full_run/splits/phantom_4000_standardization_stats.json)

### 3. Run baseline comparisons

```powershell
python code\detector\run_feature_baselines.py `
  --train data\full_run\splits\phantom_4000_train_standardized.csv `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results `
  --prefix phantom_4000
```

This produces baseline reports for:

- token only
- self-consistency only
- semantic entropy only
- groundedness only
- all four features
- manual weighted baseline
- random-score baseline

### 4. Train the logistic regression detector

```powershell
python code\detector\train_logreg_detector.py `
  --train data\full_run\splits\phantom_4000_train_standardized.csv `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results `
  --prefix phantom_4000 `
  --tune
```

This computes:

- AUROC
- AUPRC
- accuracy
- precision
- recall
- F1

It also saves the tuned detector report:

- [phantom_4000_logreg_tuned_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_4000_logreg_tuned_report.json)

### 5. Compare calibration methods

```powershell
python code\detector\compare_calibration_methods.py `
  --tuned-report results\phantom_4000_logreg_tuned_report.json `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results\calibration_compare `
  --prefix phantom_4000
```

This compares:

- raw detector probabilities
- temperature scaling
- Platt scaling
- isotonic regression

### 6. Final calibration and abstention on PHANTOM

PHANTOM uses **Platt scaling** as the final calibrator.

```powershell
python code\detector\calibrate_and_abstain.py `
  --tuned-report results\phantom_4000_logreg_tuned_report.json `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --standardization-stats data\full_run\splits\phantom_4000_standardization_stats.json `
  --output-dir results\calibration `
  --prefix phantom_4000 `
  --source-name phantom `
  --calibration-method platt
```

This computes:

- ECE
- Brier score
- NLL
- risk-coverage curve
- accuracy-coverage curve
- selected validation operating point
- frozen abstention threshold

It also saves:

- [phantom_4000_calibration_abstention_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_calibration_abstention_report.json)
- [phantom_4000_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/calibration/phantom_4000_frozen_bundle.json)

### 7. Generate PHANTOM detector plots

```powershell
python code\detector\generate_report_plots.py `
  --baseline-report results\phantom_4000_baseline_report.json `
  --tuned-report results\phantom_4000_logreg_tuned_report.json `
  --train data\full_run\splits\phantom_4000_train_standardized.csv `
  --val data\full_run\splits\phantom_4000_val_standardized.csv `
  --test data\full_run\splits\phantom_4000_test_standardized.csv `
  --output-dir results\plots `
  --prefix phantom_4000
```

## Train and evaluate the WikiQA standalone pipeline

### 1. Split the WikiQA feature table

```powershell
python code\data_gen\split_feature_table.py `
  --input data\wiki_qa\train\wikiqa_1300_feature_table.csv `
  --output-dir data\wiki_qa\splits `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --prefix wikiqa_1300
```

### 2. Fit training-set z-score stats and transform train, validation, and test

```powershell
python code\data_gen\standardize_feature_splits.py `
  --train data\wiki_qa\splits\wikiqa_1300_train.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test.csv `
  --output-dir data\wiki_qa\splits `
  --prefix wikiqa_1300
```

### 3. Run baseline comparisons

```powershell
python code\detector\run_feature_baselines.py `
  --train data\wiki_qa\splits\wikiqa_1300_train_standardized.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa `
  --prefix wikiqa_1300
```

### 4. Train the logistic regression detector

```powershell
python code\detector\train_logreg_detector.py `
  --train data\wiki_qa\splits\wikiqa_1300_train_standardized.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa `
  --prefix wikiqa_1300 `
  --tune
```

### 5. Compare calibration methods

```powershell
python code\detector\compare_calibration_methods.py `
  --tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa\calibration_compare `
  --prefix wikiqa_1300
```

### 6. Final calibration and abstention on WikiQA

WikiQA uses **isotonic regression** as the final calibrator.

```powershell
python code\detector\calibrate_and_abstain.py `
  --tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --standardization-stats data\wiki_qa\splits\wikiqa_1300_standardization_stats.json `
  --output-dir results\wikiqa\calibration `
  --prefix wikiqa_1300 `
  --source-name wikiqa `
  --calibration-method isotonic
```

This saves:

- [wikiqa_1300_calibration_abstention_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration/wikiqa_1300_calibration_abstention_report.json)
- [wikiqa_1300_frozen_bundle.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa/calibration/wikiqa_1300_frozen_bundle.json)

### 7. Generate WikiQA detector plots

```powershell
python code\detector\generate_report_plots.py `
  --baseline-report results\wikiqa\wikiqa_1300_baseline_report.json `
  --tuned-report results\wikiqa\wikiqa_1300_logreg_tuned_report.json `
  --train data\wiki_qa\splits\wikiqa_1300_train_standardized.csv `
  --val data\wiki_qa\splits\wikiqa_1300_val_standardized.csv `
  --test data\wiki_qa\splits\wikiqa_1300_test_standardized.csv `
  --output-dir results\wikiqa\plots `
  --prefix wikiqa_1300
```

## Transfer experiment 1

**Train on PHANTOM train, calibrate on PHANTOM validation, choose threshold on PHANTOM validation, freeze detector plus calibration plus threshold, test on WikiQA.**

The source-side PHANTOM training and freezing steps are the PHANTOM standalone steps above.

### 1. Apply frozen PHANTOM standardization stats to WikiQA

Use the raw WikiQA feature table. Do not standardize transfer targets with WikiQA-fitted statistics for this experiment.

```powershell
python code\data_gen\apply_standardization_stats.py `
  --stats-json results\calibration\phantom_4000_frozen_bundle.json `
  --inputs data\wiki_qa\train\wikiqa_1300_feature_table.csv `
  --output-dir data\wiki_qa\train
```

This creates:

- [wikiqa_1300_feature_table_standardized.csv](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/wiki_qa/train/wikiqa_1300_feature_table_standardized.csv)

### 2. Evaluate the frozen PHANTOM bundle on WikiQA

```powershell
python code\detector\evaluate_frozen_bundle.py `
  --bundle results\calibration\phantom_4000_frozen_bundle.json `
  --test data\wiki_qa\train\wikiqa_1300_feature_table_standardized.csv `
  --output-dir results\phantom_to_wikiqa_transfer `
  --prefix phantom_to_wikiqa `
  --source-name phantom `
  --target-name wikiqa
```

This reports:

- AUROC
- AUPRC
- accuracy
- precision
- recall
- F1
- ECE
- Brier
- risk-coverage
- accuracy-coverage

Outputs include:

- [phantom_to_wikiqa_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_transfer_report.json)
- [phantom_to_wikiqa_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_reliability_diagram.png)
- [phantom_to_wikiqa_risk_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_risk_coverage_curve.png)
- [phantom_to_wikiqa_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/phantom_to_wikiqa_transfer/phantom_to_wikiqa_accuracy_coverage_curve.png)

## Transfer experiment 2

**Train on WikiQA train, calibrate on WikiQA validation, choose threshold on WikiQA validation, freeze detector plus calibration plus threshold, test on PHANTOM.**

The source-side WikiQA training and freezing steps are the WikiQA standalone steps above.

### 1. Apply frozen WikiQA standardization stats to raw PHANTOM features

Use the raw PHANTOM feature table. Do not use PHANTOM-fitted standardization for this transfer target.

```powershell
python code\data_gen\apply_standardization_stats.py `
  --stats-json results\wikiqa\calibration\wikiqa_1300_frozen_bundle.json `
  --inputs data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped.csv `
  --output-dir data\full_run
```

This creates:

- [qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/data/full_run/qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv)

### 2. Evaluate the frozen WikiQA bundle on PHANTOM

```powershell
python code\detector\evaluate_frozen_bundle.py `
  --bundle results\wikiqa\calibration\wikiqa_1300_frozen_bundle.json `
  --test data\full_run\qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv `
  --output-dir results\wikiqa_to_phantom_transfer `
  --prefix wikiqa_to_phantom `
  --source-name wikiqa `
  --target-name phantom
```

This reports:

- AUROC
- AUPRC
- accuracy
- precision
- recall
- F1
- ECE
- Brier
- risk-coverage
- accuracy-coverage

Outputs include:

- [wikiqa_to_phantom_transfer_report.json](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_transfer_report.json)
- [wikiqa_to_phantom_reliability_diagram.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_reliability_diagram.png)
- [wikiqa_to_phantom_risk_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_risk_coverage_curve.png)
- [wikiqa_to_phantom_accuracy_coverage_curve.png](C:/Users/chinm/Documents/coursework/5541/project/Risk-Adjusted-Hallucination-Detection/results/wikiqa_to_phantom_transfer/wikiqa_to_phantom_accuracy_coverage_curve.png)

## Deeper transfer diagnostics

To compute the deeper transfer analysis used in the final writeup:

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

## Main result summary

The final story supported by this repo is:

- the four-feature detector works reasonably well in-domain
- PHANTOM is the strongest standalone result
- WikiQA is weaker and uses a different final calibrator
- calibration and abstention help most on PHANTOM
- both transfer directions are poor
- groundedness is the most stable feature across datasets
- uncertainty features are more regime-dependent

## Notes

- Older text in the repo may refer to HalluLens. The final second dataset in the completed pipeline is **WikiQA**.
- PHANTOM uses **Platt scaling** as the final calibration method.
- WikiQA uses **isotonic regression** as the final calibration method.
- The code is grounded in paper ideas, but it should be described as a practical research implementation, not an exact reproduction package.
