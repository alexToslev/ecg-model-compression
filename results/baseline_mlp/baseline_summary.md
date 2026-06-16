# Baseline MLP Summary

## What was run

A from-scratch MLP was trained on preprocessed MIT-BIH heartbeat segments. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.9316
- Final validation accuracy: 0.9296
- Test accuracy: 0.9381
- Test loss: 0.2573
- Trainable parameters: 16517
- Saved Keras model: `results/baseline_mlp/baseline_mlp.keras`

## Per-class observations

- Class 0: support=17513, precision=0.9645, recall=0.9824, f1=0.9734
- Class 1: support=285, precision=0.0000, recall=0.0000, f1=0.0000
- Class 2: support=1126, precision=0.6259, recall=0.5542, f1=0.5878
- Class 3: support=5, precision=0.0000, recall=0.0000, f1=0.0000
- Class 4: support=2078, precision=0.8646, recall=0.9038, f1=0.8838

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 93.81%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.0000).

The strongest recall is class 0 (0.9824). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[17205     0   307     0     1]
 [  229     0    56     0     0]
 [  209     0   624     0   293]
 [    5     0     0     0     0]
 [  190     0    10     0  1878]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Run MLP pruning and manual 8-bit quantization, then compare accuracy, sparsity, and model size against this baseline.
