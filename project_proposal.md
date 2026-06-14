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

### WP1 — Project setup and planning

Objective: establish the project structure, branch workflow, and deliverables.

- WP1.1: design repository layout and write `project_proposal.md`.
- WP1.2: add README usage instructions and a professional CLI guidance.
- WP1.3: create the branch strategy and commit mapping for `WPx.y` subtasks.

Assigned primary owner: Student 1

### WP2 — ECG dataset preprocessing and visualization

Objective: implement the ECG data pipeline from scratch with loading, normalization, splitting, and plots.

- WP2.1: implement `src/data/mitbih_csv.py` for CSV loading and dataset validation.
- WP2.2: add `src/data/visualization.py` for signal plotting, class balance, and split diagnostics.
- WP2.3: implement normalization options: `none`, `standard`, and `per_sample`.
- WP2.4: add synthetic demo dataset generation for development and debugging.
- WP2.5: write dataset diagnostics that confirm sample counts, input shape, and class coverage.

Assigned primary owner: Student 1

### WP3 — Baseline CNN architecture and training

Objective: build a from-scratch 1D CNN model and training pipeline that can run on demo or full ECG data.

- WP3.1: implement `src/models/cnn1d.py` with explicit Conv1D, ReLU, MaxPool1D, Flatten, and Dense layers, plus manual forward/backward propagation and parameter updates.
- WP3.2: implement `src/train_cnn.py` with a manual training loop, batch updates, validation metrics, history logging, and output artifact saving.
- WP3.3: support reproducibility with random seeds and reproducible dataset splits.
- WP3.4: provide both demo mode and full dataset mode in training.

Assigned primary owner: Student 2

### WP4 — Testing and evaluation

Objective: implement full evaluation on hold-out test data with reports and visual summaries.

- WP4.1: implement `src/evaluate_model.py` for float32 model testing.
- WP4.2: implement `src/evaluation/summarize_baseline.py` for plots and markdown summaries.
- WP4.3: add prediction reporting, confusion matrix, and class metrics.
- WP4.4: include model size and parameter count in the reporting.

Assigned primary owner: Student 2

### WP5 — Structured quantization and compression

Objective: add TFLite int8 conversion and compare accuracy and model size.

- WP5.1: implement `src/compression/quantize_tflite.py` with representative dataset support.
- WP5.2: evaluate the quantized model for accuracy and file size.
- WP5.3: document compression results and comparison metrics.
- WP5.4: prepare reproducible commands for compression experiments.

Assigned primary owner: Student 2

### WP6 — Pruning, analysis, and final report

Objective: complete pruning experiments, compare results, and publish final documentation.

- WP6.1: add pruning pipeline and retraining for compressed models.
- WP6.2: compare pruned vs. quantized vs. baseline accuracy and size.
- WP6.3: prepare final report and lessons learned.
- WP6.4: validate commit history and merge into the shared branch.

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