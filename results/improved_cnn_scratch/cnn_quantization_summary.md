# CNN Post-Training Quantization Summary

This report compares the improved CNN Keras model against a post-training int8 TensorFlow Lite model.

## Quantized model

- Int8 accuracy: 0.8787
- Int8 loss: 0.4264
- TFLite size: 23576 bytes
- TFLite path: `results\improved_cnn_scratch\tiny_ecg_cnn_int8.tflite`
- Representative samples: 200

## Baseline vs int8

| metric | baseline CNN | int8 TFLite CNN | change |
|---|---:|---:|---:|
| accuracy | 0.8736 | 0.8787 | +0.0051 |
| loss | 0.4070 | 0.4264 | +0.0194 |
| size bytes | 90416 | 23576 | -66840 |

- Size reduction: 73.92%

Generated plots:

- `cnn_quantization_accuracy_comparison.png`
- `cnn_quantization_size_comparison.png`

Generated evaluation artifacts:

- `int8_metrics.json`
- `int8_classification_report.json`
- `int8_confusion_matrix.csv`
