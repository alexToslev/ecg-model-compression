# ECG CNN Improvements

This branch focuses only on improving the 1D CNN for MIT-BIH heartbeat classification.

The goal is to address the main weakness seen in earlier experiments: high overall accuracy can hide poor recall for rare heartbeat classes, especially class 1 and class 3.

## Dataset

Use the preprocessed MIT-BIH heartbeat CSV files:

```text
data/processed/mitbih_train.csv
data/processed/mitbih_test.csv
```

Each row contains 187 ECG values and the class label in the final column.

The PTBDB files from the Kaggle dataset are not mixed into this branch because they are a different binary normal/abnormal task. This branch keeps the MIT-BIH 5-class task consistent.

## Improved CNN

The improved model uses:

- Conv1D 16 filters, kernel size 7
- BatchNorm, ReLU, MaxPool
- Conv1D 32 filters, kernel size 5
- BatchNorm, ReLU, MaxPool
- Conv1D 64 filters, kernel size 3
- BatchNorm, ReLU
- GlobalAveragePooling1D
- Dense 32
- Dropout
- Dense 5 with softmax

Training improvements:

- Adam optimizer
- class-weighted loss for rare classes
- optional rare-class ECG augmentation
- early stopping
- learning-rate reduction on plateau
- macro F1 and weighted F1 saved in metrics

## Setup

```bash
pip install -r requirements.txt
python -m src --help
```

## Train Improved CNN

Recommended first run:

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

For a quick smoke test without the real dataset:

```bash
python -m src train --demo-data --epochs 1 --output-dir results/smoke_improved_cnn
```

## Evaluate

```bash
python -m src evaluate \
  --model results/improved_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed
```

## Quantize To Int8 TensorFlow Lite

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

## ESP32 Export Planning

After quantization:

```bash
python -m src export-tinyml \
  --model results/improved_cnn/tiny_ecg_cnn_int8.tflite \
  --metrics results/improved_cnn/int8_metrics.json \
  --output-dir results/esp32
```

## What To Check

Do not judge only by total accuracy. Check:

- class 1 recall
- class 3 recall
- macro F1
- confusion matrix
- int8 accuracy after quantization
- model size and ESP32 memory estimate

The expected tradeoff is that class weighting may reduce class 0 accuracy a little, but should improve rare-class recall if the model learns useful minority-class patterns.
