# Manual MLP Quantization Summary

This report compares the original manual MLP against an 8-bit fixed-point quantized version.

## Original vs Quantized

| metric | original float32 MLP | manual int8 MLP | change |
|---|---:|---:|---:|
| test accuracy | 0.9381 | 0.9378 | -0.0003 |
| test loss | 0.2573 | 0.2571 | -0.0001 |
| model size bytes | 66068 | 16940 | -49128 |

## Compression

- Compression ratio: 3.9001x
- Size reduction: 74.36%
- Prediction agreement with original model: 0.9968
- Calibration samples: 200
- Parameters: 16517

## Generated plots

- `quantization_accuracy_comparison.png`
- `quantization_loss_comparison.png`
- `quantization_size_comparison.png`

## Generated comparison artifacts

- `original_vs_quantized_metrics.csv`
- `original_classification_report.json`
- `quantized_classification_report.json`
- `original_confusion_matrix.csv`
- `quantized_confusion_matrix.csv`

## Notes

- The quantized model uses symmetric int8 weights and int32 bias accumulation.
- Fixed-point inference uses calibrated input and hidden activation scales.