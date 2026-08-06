# From-Scratch Improved CNN Summary

## What was run

A manual NumPy CNN was trained with explicit forward pass, backward pass, weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.1872
- Final validation accuracy: 0.1702
- Test accuracy: 0.1695
- Test loss: 1.9191
- Trainable parameters: 11173
- Saved Keras model: `results/sampling_sweep_cap_4/weighted/tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=18118, precision=0.9865, recall=0.0040, f1=0.0080
- Class 1: support=556, precision=0.0374, recall=0.9263, f1=0.0718
- Class 2: support=1448, precision=0.3607, recall=0.9572, f1=0.5239
- Class 3: support=162, precision=0.1058, recall=0.8951, f1=0.1892
- Class 4: support=1608, precision=0.5638, recall=0.9894, f1=0.7183

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 16.95%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 0 has the weakest recall (0.0040).

The strongest recall is class 4 (0.9894). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[   73 13236  2416  1187  1206]
 [    0   515    22    14     5]
 [    1    18  1386    24    19]
 [    0     2    14   145     1]
 [    0    11     5     1  1591]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.
