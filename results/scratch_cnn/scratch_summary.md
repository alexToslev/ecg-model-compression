# From-Scratch CNN Summary

This run trains a small 1D CNN with manual NumPy code for the learning parts.

Implemented from scratch:

- Conv1D forward and backward pass
- ReLU forward and backward pass
- MaxPool1D forward and backward pass
- Global average pooling backward pass
- Dense layer forward and backward pass
- Softmax cross-entropy loss and gradient
- Mini-batch SGD weight updates

This branch is for learning and comparison. The TensorFlow/Keras branch remains the practical path for TensorFlow Lite and ESP32 deployment.

## Results

- Test accuracy: 0.8280
- Parameters: 173
- Training seconds: 392.93
- Max train samples: 0
