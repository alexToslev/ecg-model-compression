# CNN quantization workflow

WP8 compares the from-scratch baseline CNN against post-training int8 quantization and quantization-aware training (QAT). The scratch NumPy CNN remains the training implementation; Keras is used only as an export format for TensorFlow Lite conversion.

## WP8.1 Post-training quantization

Run post-training quantization from the saved baseline CNN Keras export:

```bash
python -m src quantize \
  --model results/baseline_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed \
  --output results/baseline_cnn/tiny_ecg_cnn_int8.tflite
```

This writes:

- `results/baseline_cnn/tiny_ecg_cnn_int8.tflite`
- `results/baseline_cnn/int8_metrics.json`
- `results/baseline_cnn/int8_classification_report.json`
- `results/baseline_cnn/int8_confusion_matrix.csv`
- `results/baseline_cnn/cnn_quantization_summary.md`
- `results/baseline_cnn/cnn_quantization_accuracy_comparison.png`
- `results/baseline_cnn/cnn_quantization_size_comparison.png`

The int8 report compares accuracy, loss, and file size against `results/baseline_cnn/metrics.json` when that baseline metrics file is available.

## WP8.2 Quantization-aware CNN training

Run QAT with fake-quantized weights and activations inside the scratch CNN:

```bash
python -m src train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 128 \
  --quantize-aware \
  --output-dir results/baseline_cnn_qat
```

The QAT path uses the same from-scratch layers as the baseline CNN. During training it fake-quantizes the forward activations and requantizes Conv1D and Dense weights after each update. Validation and test metrics are also computed through the fake-quantized forward path so the reported accuracy reflects quantized inference behavior.

The command writes the same training artifacts as the baseline CNN, including `metrics.json`, `history.csv`, `classification_report.json`, `confusion_matrix.csv`, and plots under `results/baseline_cnn_qat/`.

## WP8.3 Compression comparison

After baseline training, structured pruning, post-training quantization, and QAT are available, generate a single CNN comparison table and plots:

```bash
python -m src compare-cnn \
  --baseline-dir results/baseline_cnn \
  --pruning-dir results/baseline_cnn/pruning \
  --quantized-dir results/baseline_cnn \
  --qat-dir results/baseline_cnn_qat \
  --output-dir results/baseline_cnn/quantization
```

This writes:

- `results/baseline_cnn/quantization/cnn_compression_comparison.csv`
- `results/baseline_cnn/quantization/cnn_compression_comparison.md`
- `results/baseline_cnn/quantization/cnn_compression_accuracy.png`
- `results/baseline_cnn/quantization/cnn_compression_size.png`
- `results/baseline_cnn/quantization/cnn_compression_accuracy_vs_size.png`

The comparison includes baseline CNN, the best available nonzero structured-pruned CNN result, post-training int8 TFLite CNN, and QAT CNN if `--qat-dir` is provided.

## Interpretation notes

- Post-training quantization reports the actual `.tflite` file size.
- QAT currently reports the saved Keras/scratch export size unless it is also converted to TFLite afterward.
- Structured-pruned CNN size is estimated from remaining nonzero float32 parameters unless the architecture is physically shrunk or stored in a sparse format.
- Do not mix demo-data QAT results with real MIT-BIH baseline results in the final comparison. Demo commands are useful only for smoke testing the pipeline.

## Smoke-test commands

Fast checks without the full MIT-BIH run:

```bash
python -m src train --demo-data --epochs 1 --quantize-aware --output-dir /tmp/ecg_wp8_cnn_qat
python -m src quantize --demo-data --model /tmp/ecg_wp8_cnn_qat/tiny_ecg_cnn.keras --output /tmp/ecg_wp8_cnn_qat/tiny_ecg_cnn_int8.tflite
python -m src compare-cnn --baseline-dir results/baseline_cnn --pruning-dir results/baseline_cnn/pruning --quantized-dir results/baseline_cnn --output-dir /tmp/ecg_wp8_comparison
```
