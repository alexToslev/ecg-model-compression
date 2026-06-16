# MLP Quantization Workflow

This document describes WP5: manual 8-bit MLP quantization and comparison
against the original float32 MLP.

## Goal

The goal is to show how much model size can be reduced by quantizing the MLP
while measuring the accuracy and loss change on the same test set.

The primary comparison is:

- original float32 manual MLP
- manual int8 post-training quantized MLP

The project also supports quantization-aware MLP training as an extra
comparison run.

## Post-Training Quantization Command

First train the baseline MLP:

```bash
python -m src train-mlp --data-dir data/processed --epochs 20 --output-dir results/baseline_mlp
```

Then quantize and compare it:

```bash
python -m src quantize-mlp --weights results/baseline_mlp/baseline_mlp_weights.npz --data-dir data/processed --output-dir results/baseline_mlp/quantization
```

Fast demo-data smoke command:

```bash
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_mlp_quant_smoke
python -m src quantize-mlp --demo-data --weights /tmp/ecg_mlp_quant_smoke/baseline_mlp_weights.npz --output-dir /tmp/ecg_mlp_quant_smoke/quantization
```

## Output Artifacts

The quantization workflow writes:

- `baseline_mlp_quantized.npz`
- `quantization_metrics.json`
- `quantization_metrics.csv`
- `original_vs_quantized_metrics.csv`
- `quantization_summary.md`
- `quantization_accuracy_comparison.png`
- `quantization_loss_comparison.png`
- `quantization_size_comparison.png`
- `original_classification_report.json`
- `quantized_classification_report.json`
- `original_confusion_matrix.csv`
- `quantized_confusion_matrix.csv`

The most important report file is `quantization_summary.md`, because it gives a
compact table with original accuracy/loss/size, quantized accuracy/loss/size,
and the measured deltas.

## Metrics to Use in the Final Report

Use these fields from `quantization_metrics.json`:

- `original_test_accuracy`
- `quantized_test_accuracy`
- `accuracy_delta`
- `original_test_loss`
- `quantized_test_loss`
- `loss_delta`
- `original_model_size_bytes`
- `quantized_model_size_bytes`
- `compression_ratio`
- `size_reduction_percent`
- `prediction_agreement`

The quantized size is computed from int8 weights, int32 biases, and scale
metadata. This is a model-representation estimate for the manual NumPy
quantized model.

## Quantization-Aware Training

The MLP training command can simulate quantization during training:

```bash
python -m src train-mlp --data-dir data/processed --epochs 20 --quantize-aware --output-dir results/baseline_mlp_qat
```

Demo-data smoke command:

```bash
python -m src train-mlp --demo-data --epochs 1 --quantize-aware --output-dir /tmp/ecg_mlp_qat_smoke
```

QAT runs include these metric fields:

- `quantize_aware_training`
- `training_mode`

This prevents QAT output folders from being confused with the normal float32
baseline.

## Interpretation Notes

Post-training quantization is applied after training. It is simple and usually
gives a strong size reduction, but accuracy can drop because weights and
activations are rounded to int8.

Quantization-aware training exposes the model to fake quantization during
training. It can reduce the accuracy gap, but it is a separate training run and
should be compared fairly against both the float32 baseline and the
post-training quantized model.
