# CNN Post-Training Quantization Summary

This report compares the exported scratch CNN Keras model against a post-training int8 TensorFlow Lite model.

## Quantized model

- Int8 accuracy: 0.8749
- Int8 loss: 0.4586
- TFLite size: 62856 bytes
- TFLite path: `results/baseline_cnn/tiny_ecg_cnn_int8.tflite`
- Representative samples: 200

## Baseline vs int8

| metric | baseline CNN | int8 TFLite CNN | change |
|---|---:|---:|---:|
| accuracy | 0.8753 | 0.8749 | -0.0004 |
| loss | 0.3835 | 0.4586 | +0.0750 |
| size bytes | 247144 | 62856 | -184288 |

- Size reduction: 74.57%

Generated plots:

- `cnn_quantization_accuracy_comparison.png`
- `cnn_quantization_size_comparison.png`

Generated evaluation artifacts:

- `int8_metrics.json`
- `int8_classification_report.json`
- `int8_confusion_matrix.csv`
