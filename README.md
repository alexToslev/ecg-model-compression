# ecg-model-compression
From-scratch compression of neural networks for ECG classification using pruning, quantization, and ESP32 deployment.

## First baseline: tiny 1D CNN

This first version uses the common preprocessed MIT-BIH heartbeat CSV format:

- `data/processed/mitbih_train.csv`
- `data/processed/mitbih_test.csv`

Each row should contain one heartbeat segment, with the class label in the final column.
The popular Kaggle "Heartbeat Categorization Dataset" provides this format.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a quick smoke test with synthetic ECG-like data:

```bash
python -m src.train_cnn --demo-data --epochs 3
python -m src.compression.quantize_tflite --demo-data
```

Train on the real MIT-BIH CSV files:

```bash
python -m src.train_cnn --data-dir data/processed --epochs 20
```

Convert the trained model to int8 TensorFlow Lite:

```bash
python -m src.compression.quantize_tflite --data-dir data/processed
```

This creates a full quantization comparison in `results/quantized/`:

- `tiny_ecg_cnn_int8.tflite` - full-int8 TensorFlow Lite model
- `quantization_metrics.json` - accuracy, file size, and quantization details
- `float32_vs_int8.csv` - simple float32/int8 comparison table
- `float32_classification_report.json` - per-class float32 metrics
- `int8_classification_report.json` - per-class int8 metrics
- `int8_confusion_matrix.csv` - int8 confusion matrix
- `quantization_summary.md` - short report-style explanation and results

If you want one command for a quick synthetic-data quantization smoke test, use:

```bash
python -m src.compression.quantize_tflite --demo-data --train-if-missing --epochs 3
```

## Quantization explanation

Post-training int8 quantization converts a trained float32 model after training. TensorFlow Lite uses representative ECG samples to estimate numeric ranges, then stores values with an integer, scale, and zero-point:

```text
real_value = scale * (integer_value - zero_point)
```

Intuitively, this is like measuring the same ECG signal with a coarser ruler. The model becomes smaller and more suitable for microcontrollers, but rounding can slightly change predictions. Therefore, the int8 model must be evaluated separately from the float32 Keras model.

For MIT-BIH, total accuracy is not enough. Because the dataset is imbalanced, always compare per-class precision, recall, F1-score, and the confusion matrix. In the earlier baseline, minority classes such as class 1 and class 3 had weak recall even though total accuracy was high.

## ESP32 feasibility checklist

Before deploying on ESP32, still check:

- final `.tflite` file size
- RAM needed for intermediate tensors
- TensorFlow Lite Micro operator support
- conversion of the `.tflite` file to a C array
- real on-device inference time and memory usage

Outputs are written under `results/`.
