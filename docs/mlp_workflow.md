# Baseline MLP Workflow

This document describes WP3: the from-scratch baseline MLP used to validate the
ECG compression pipeline before the CNN compression work.

## Purpose

The MLP is intentionally simpler than the CNN. It verifies that the project can:

- load the ECG dataset consistently
- train a manual neural network with NumPy
- save reproducible evaluation artifacts
- support later pruning and quantization experiments

This makes the MLP a useful baseline for explaining compression before moving
to the more complex 1D CNN.

## Model

The implementation is in `src/models/mlp.py`.

The model contains:

- `Flatten`
- one or more manual `Dense` layers
- `ReLU` activations
- an output `Dense` layer
- manual softmax cross-entropy loss
- explicit forward, backward, and update methods

The default training configuration uses two hidden dense layers with 64 units
each.

## Training Command

Fast smoke test with demo data:

```bash
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_mlp_smoke
```

Full dataset baseline:

```bash
python -m src train-mlp --data-dir data/processed --epochs 20 --output-dir results/baseline_mlp
```

Optional quantization-aware MLP training:

```bash
python -m src train-mlp --data-dir data/processed --epochs 20 --quantize-aware --output-dir results/baseline_mlp_qat
```

Use the same `--normalize`, `--validation-fraction`, and `--seed` values when
comparing this baseline against pruned or quantized variants.

## Output Artifacts

The baseline MLP training command writes:

- `baseline_mlp_weights.npz`
- `baseline_mlp.keras`
- `history.csv`
- `metrics.json`
- `classification_report.json`
- `confusion_matrix.csv`
- `baseline_summary.md`
- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

`metrics.json` records the test loss, test accuracy, input length, number of
classes, trainable parameter count, saved weights path, Keras export path, and
saved model sizes when available.

## Why This Is Reproducible

The training command accepts `--seed`, which controls model initialization and
the dataset validation split. The default seed is `42`. The train/validation
split and normalization are provided by the shared WP2 dataset loader.

## Next Work Packages

The MLP baseline feeds directly into:

- WP4: magnitude-based and structured MLP pruning
- WP5: manual fixed-point MLP quantization and MLP QAT comparison
- WP9: final benchmark table across all baseline and compressed models
