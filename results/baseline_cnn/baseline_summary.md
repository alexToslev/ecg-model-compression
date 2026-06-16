# Baseline 1D CNN Summary

## What was run

A small 1D CNN was trained on preprocessed MIT-BIH heartbeat segments. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.9436
- Final validation accuracy: 0.9429
- Test accuracy: 0.8753
- Test loss: 0.3835
- Trainable parameters: 49781
- Saved Keras model: `results/baseline_cnn/tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=17513, precision=0.9670, recall=0.9199, f1=0.9429
- Class 1: support=285, precision=0.0000, recall=0.0000, f1=0.0000
- Class 2: support=1126, precision=0.1703, recall=0.2700, f1=0.2089
- Class 3: support=5, precision=0.0000, recall=0.0000, f1=0.0000
- Class 4: support=2078, precision=0.7998, recall=0.9495, f1=0.8682

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 87.53%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.0000).

The strongest recall is class 4 (0.9495). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[16111     5  1390     0     7]
 [  195     0    90     0     0]
 [  328     0   304     7   487]
 [    5     0     0     0     0]
 [   22     0     1    82  1973]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Run structured CNN pruning and int8 TensorFlow Lite quantization, then compare accuracy and model size against this float32 baseline.
