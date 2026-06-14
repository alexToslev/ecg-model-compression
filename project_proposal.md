# ECG Model Compression Project Proposal

## Project title

Compressed 1D CNN ECG Classification with Pruning, Quantization, and Evaluation

## Team members

- Student 1: [Your Name]
- Student 2: [Colleague Name]

## Project overview

This project delivers a from-scratch ECG heartbeat classification pipeline with a lightweight 1D CNN, reproducible dataset handling, and compression experiments. The aim is to demonstrate how pruning and TensorFlow Lite int8 quantization reduce model size while preserving classification performance.

## Problem statement

ECG classification models are valuable for arrhythmia detection, but medical devices and embedded targets require small, efficient models. This project proves that compression techniques can preserve accuracy while making ECG models deployable on constrained hardware.

## Objectives

1. Build a reproducible ECG data pipeline for preprocessed MIT-BIH heartbeat CSV data.
2. Implement a strong baseline 1D CNN classifier.
3. Create automated evaluation and reporting for accuracy, confusion matrices, and class metrics.
4. Develop a compression pipeline with TensorFlow Lite int8 quantization.
5. Add pruning-based compression and compare results against the baseline.
6. Document the full workflow, results, and reproducible experiment commands.

## Dataset

- Preprocessed MIT-BIH heartbeat CSV dataset with 187 ECG signal values per heartbeat and a class label in the final column.
- Expected files:
  - `data/processed/mitbih_train.csv`
  - `data/processed/mitbih_test.csv`

## Branch and commit workflow

- Work in branch `cnn_implementation`.
- Commit each completed subtask with a clear prefix: `WP1.1`, `WP1.2`, ..., `WP5.2`.
- Use commit messages such as `WP1.1: add baseline data loader`, `WP2.1: implement evaluation CLI`, `WP3.2: add int8 quantization metrics`.
- This makes individual contributions explicit and allows professors to trace who implemented each feature.

## Work packages

### WP1 — Baseline pipeline and project setup

Objective: establish the core repository structure, dataset loader, and baseline training flow.

- WP1.1: design repository layout and write `project_proposal.md`.
- WP1.2: implement `src/data/mitbih_csv.py` with dataset validation, preprocessing, and demo data support. Use the smaller synthetic demo dataset for initial development while keeping the full MIT-BIH CSV workflow ready for later stages.
- WP1.3: implement `src/models/cnn1d.py` with a small 1D CNN architecture.
- WP1.4: implement `src/train_cnn.py` for training, saving model artifacts, metrics, and history.
- WP1.5: add README usage instructions and baseline CLI guidance.

Assigned primary owner: Student 1

### WP2 — Evaluation and reporting

Objective: build evaluation tooling, plots, and summary generation for the baseline model.

- WP2.1: implement `src/evaluate_model.py` for float32 model evaluation on test data.
- WP2.2: implement `src/evaluation/summarize_baseline.py` for learning curves, confusion matrix, and class metrics.
- WP2.3: add evaluation reporting outputs: `metrics.json`, `classification_report.json`, `history.csv`, plots, and summary markdown.
- WP2.4: update documentation with `python -m src evaluate` and `python -m src summarize` commands.

Assigned primary owner: Student 1

### WP3 — Quantization pipeline

Objective: add a reproducible TFLite int8 quantization pipeline and compare compressed model accuracy and size.

- WP3.1: implement `src/compression/quantize_tflite.py` with representative dataset generation and int8 conversion.
- WP3.2: add int8 evaluation and size reporting for the quantized model.
- WP3.3: integrate the compression pipeline into the unified CLI or documentation.
- WP3.4: document expected quantized outputs and comparison metrics.

Assigned primary owner: Student 2

### WP4 — Pruning and advanced compression

Objective: extend the project with pruning experiments and model size/accuracy tradeoff analysis.

- WP4.1: implement pruning-aware training or weight pruning for the 1D CNN.
- WP4.2: add model checkpoints for pruned and retrained compressed models.
- WP4.3: evaluate pruned model accuracy and compare size vs. baseline.
- WP4.4: document pruning results in markdown and add comparison tables.

Assigned primary owner: Student 2

### WP5 — Final report and reproducibility

Objective: finalize documentation, validate the workflow, and prepare the repository for grading.

- WP5.1: produce a final written report with methodology, results, and lessons learned.
- WP5.2: create a reproducible command list and summary of all experiments.
- WP5.3: validate the branch history and ensure each commit corresponds to a WP subtask.
- WP5.4: merge changes into the shared branch after both students finish their parts.

Assigned owners: both students

## Success criteria

- The baseline model trains correctly on demo and real ECG data.
- The evaluation pipeline creates reliable metrics, plots, and summary files.
- The quantized int8 model is generated and evaluated successfully.
- The pruning pipeline produces a smaller model with acceptable accuracy.
- The repository includes professional documentation and clear commit history.

## Expected commands

- `python -m src train --data-dir data/processed --epochs 20`
- `python -m src evaluate --model results/baseline_cnn/tiny_ecg_cnn.keras --data-dir data/processed`
- `python -m src quantize --model results/baseline_cnn/tiny_ecg_cnn.keras --data-dir data/processed`
- `python -m src summarize --run-dir results/baseline_cnn`

## Timeline estimate

Each student contributes at least 120 hours individually.

- WP1: Baseline pipeline and setup — 30 hours each
- WP2: Evaluation and reporting — 30 hours each
- WP3: Quantization and compression — 30 hours each
- WP4: Pruning experiments and analysis — 20 hours each
- WP5: Reporting, validation, and merge — 10 hours each

This structure allows clear ownership, separate commit history, and a professional grading trail.