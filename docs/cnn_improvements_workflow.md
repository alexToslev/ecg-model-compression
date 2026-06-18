# CNN Improvements Workflow

This branch focuses on improving the CNN classifier for the imbalanced MIT-BIH 5-class heartbeat task.

It intentionally removes the MLP and pruning workflows from the active code path. The aim is to test whether a better CNN training setup can improve minority-class recall before doing another ESP32 deployment test.

## Problem

The earlier models showed good overall accuracy, but rare heartbeat classes were weak. In particular, class 1 and class 3 had very low or zero recall in some runs.

Overall accuracy is not enough for this dataset because class 0 is much more common than the other classes.

## Model Structure

The improved CNN is:

```text
Input 187 x 1
Conv1D 16 filters, kernel 7
BatchNorm
ReLU
MaxPool1D
Conv1D 32 filters, kernel 5
BatchNorm
ReLU
MaxPool1D
Conv1D 64 filters, kernel 3
BatchNorm
ReLU
GlobalAveragePooling1D
Dense 32
Dropout
Dense 5 softmax
```

Why this structure:

- Conv1D layers learn local ECG waveform patterns.
- Increasing filters from 16 to 32 to 64 lets deeper layers learn richer heartbeat features.
- BatchNorm makes training more stable.
- GlobalAveragePooling keeps the model smaller than a large Flatten layer.
- Dropout helps reduce overfitting after rare-class augmentation.

## Imbalance Handling

Two mechanisms are used.

### Class-Weighted Loss

The loss function gives larger penalty to mistakes on rare classes. This forces the model to care more about class 1 and class 3 instead of mostly optimizing class 0.

### Rare-Class Augmentation

Rare classes get mild synthetic variations:

- small noise
- small amplitude scaling
- small baseline shift
- small time shift

The augmentation is intentionally mild so the heartbeat shape is still realistic.

## Train

```bash
python -m src train \
  --data-dir data/processed \
  --output-dir results/improved_cnn \
  --epochs 30 \
  --batch-size 128 \
  --class-weights balanced \
  --augment-rare-classes \
  --rare-target-count 2000
```

For a smoke test:

```bash
python -m src train --demo-data --epochs 1 --output-dir results/smoke_improved_cnn
```

## Quantize

```bash
python -m src quantize \
  --model results/improved_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed \
  --output results/improved_cnn/tiny_ecg_cnn_int8.tflite
```

## Summarize

```bash
python -m src summarize \
  --run-dir results/improved_cnn \
  --quantized-dir results/improved_cnn
```

## What To Compare

The most important metrics are:

- macro F1
- class 1 recall
- class 3 recall
- confusion matrix
- int8 accuracy drop after quantization
- TFLite model size

Expected result: class weighting may reduce normal-beat accuracy slightly, but it should improve rare-class recall if the model learns useful minority-class patterns.
