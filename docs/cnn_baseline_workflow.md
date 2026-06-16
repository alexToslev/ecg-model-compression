# Baseline 1D CNN Workflow

This document supports WP6.2. The CNN is the target model for the project, so
the baseline workflow must stay clear and reproducible, with each computation easy to inspect.

## Model Scope

The baseline model is a small 1D CNN for ECG heartbeat classification. The core
model is implemented from scratch in `src/models/cnn1d.py`.

The manual layers are:

- `Conv1D`
- `ReLU`
- `MaxPool1D`
- `Flatten`
- `Dense`
- `SoftmaxCrossEntropy`

The model implements explicit:

- forward propagation
- loss computation
- backward propagation
- parameter updates

TensorFlow/Keras is only used for export after training so the model can later
be converted to TensorFlow Lite. The training path itself uses the scratch NumPy
model.

## Architecture

The default CNN uses:

| stage | layer |
|---|---|
| 1 | Conv1D, 8 filters, kernel size 7 |
| 2 | ReLU |
| 3 | MaxPool1D |
| 4 | Conv1D, 16 filters, kernel size 5 |
| 5 | ReLU |
| 6 | MaxPool1D |
| 7 | Conv1D, 32 filters, kernel size 3 |
| 8 | ReLU |
| 9 | MaxPool1D |
| 10 | Flatten |
| 11 | Dense, 64 hidden units |
| 12 | ReLU |
| 13 | Dense output layer |

For the standard MIT-BIH heartbeat input length of 187 and 5 classes, the model
has 49781 parameters.

## Smoke Command

Use demo data for a fast check:

```bash
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_cnn_smoke
```

This should create:

- `tiny_ecg_cnn_weights.npz`
- `tiny_ecg_cnn.keras`
- `history.csv`
- `metrics.json`
- `classification_report.json`
- `confusion_matrix.csv`
- `baseline_summary.md`
- `plots/`
- `dataset_visualizations/`

## Full Baseline Command

Run the full 20-epoch baseline with:

```bash
python -m src train --data-dir data/processed --epochs 20 --output-dir results/baseline_cnn --seed 42 --normalize none
```

Only run this command when intentionally regenerating the canonical baseline.
For experiments, use a different output directory so the preserved 20-epoch
baseline is not overwritten accidentally.

## Evaluation and Summary

Regenerate plots and summary for an existing run:

```bash
python -m src summarize --run-dir results/baseline_cnn
```

Evaluate the exported Keras model:

```bash
python -m src evaluate --model results/baseline_cnn/tiny_ecg_cnn.keras --data-dir data/processed --normalize none
```

## Why This Baseline Matters

WP7 structured pruning, WP8 CNN quantization, and WP9 benchmarking should all
compare against this baseline. The final report should treat the CNN baseline as
the target model and the MLP as a supporting validation model.
