# Quantization Summary

## Intuition

Post-training int8 quantization stores model weights and activations as 8-bit integers instead of 32-bit floating point numbers. This usually makes the model smaller and more microcontroller-friendly, but rounding can slightly change the predictions.

## Formal View

TensorFlow Lite represents a real tensor value using an integer value, a scale, and a zero-point:

`real_value = scale * (integer_value - zero_point)`

The representative dataset is used during conversion to estimate useful scales for ECG inputs and intermediate activations.

## Results

- Parameters: 2437
- Input length/classes: 187 / 5
- Float32 accuracy: 0.9354
- Int8 accuracy: 0.9342
- Accuracy drop: 0.0011
- Size: 69,963 bytes -> 11,928 bytes (5.87x smaller)
- Representative samples: 200

## ESP32 Feasibility

The `.tflite` file size is only the first check. Before deployment, also check TensorFlow Lite Micro operator support, RAM for intermediate tensors, C-array export, and real on-device inference time.

## Class Imbalance Note

For MIT-BIH, total accuracy is not enough. Always inspect the per-class reports and confusion matrix, because minority-class recall can be weak even when overall accuracy is high.
