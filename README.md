# ECG From-Scratch CNN Improvements

This branch focuses on a manual NumPy 1D CNN for MIT-BIH heartbeat classification.

This version adds practical strategies for class imbalance while keeping the NumPy implementation straightforward to inspect:

- class-weighted softmax cross-entropy
- rare-class augmentation
- optional Adam updates implemented manually
- macro F1 and per-class recall reporting

TensorFlow/Keras is used only after training to export the learned weights to a `.keras` model for later TensorFlow Lite quantization. The training itself uses explicit forward pass, backward pass, loss gradient, and weight updates.

## Dataset

Use:

```text
data/processed/mitbih_train.csv
data/processed/mitbih_test.csv
```

Each row contains 187 ECG values and one class label.

## Model Structure

```text
Input 187 x 1
Conv1D 16 filters, kernel 7
ReLU
MaxPool1D
Conv1D 32 filters, kernel 5
ReLU
MaxPool1D
Conv1D 64 filters, kernel 3
ReLU
GlobalAveragePool1D
Dense 32
ReLU
Dropout
Dense 5
Weighted Softmax Cross-Entropy
```

## Train

First try a quick smoke run:

```bash
C:/venvs/ecg/Scripts/python.exe -m src train --demo-data --epochs 1 --output-dir results/smoke_scratch_cnn
```

Full training:

```bash
C:/venvs/ecg/Scripts/python.exe -m src train --data-dir data/processed --output-dir results/improved_cnn_scratch --epochs 20 --batch-size 128 --optimizer adam --learning-rate 0.001 --class-weights balanced --class-weight-cap 10 --augment-rare-classes --rare-target-count 2000
```

If this is too slow, run a smaller first experiment:

```bash
C:/venvs/ecg/Scripts/python.exe -m src train --data-dir data/processed --output-dir results/improved_cnn_scratch_small --epochs 5 --batch-size 128 --optimizer adam --class-weights balanced --class-weight-cap 10 --augment-rare-classes --rare-target-count 1000 --max-train-samples 10000
```

## What To Inspect

```bash
C:/venvs/ecg/Scripts/python.exe -m json.tool results/improved_cnn_scratch/metrics.json
C:/venvs/ecg/Scripts/python.exe -m json.tool results/improved_cnn_scratch/classification_report.json
```

Important metrics:

- total accuracy
- macro F1
- class 1 recall and precision
- class 3 recall and precision
- confusion matrix

## Quantize Later

After training, if Keras export succeeds:

```bash
C:/venvs/ecg/Scripts/python.exe -m src quantize --model results/improved_cnn_scratch/tiny_ecg_cnn.keras --data-dir data/processed --output results/improved_cnn_scratch/tiny_ecg_cnn_int8.tflite
```
