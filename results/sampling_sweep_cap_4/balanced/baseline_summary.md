# From-Scratch Improved CNN Summary

## What was run

A manual NumPy CNN was trained with explicit forward pass, backward pass, weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.1927
- Final validation accuracy: 0.1750
- Test accuracy: 0.1749
- Test loss: 1.8431
- Trainable parameters: 11173
- Saved Keras model: `results/sampling_sweep_cap_4/balanced/tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=18118, precision=1.0000, recall=0.0113, f1=0.0224
- Class 1: support=556, precision=0.0378, recall=0.9371, f1=0.0727
- Class 2: support=1448, precision=0.4238, recall=0.9454, f1=0.5853
- Class 3: support=162, precision=0.0691, recall=0.9259, f1=0.1285
- Class 4: support=1608, precision=0.6304, recall=0.9845, f1=0.7686

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 17.49%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 0 has the weakest recall (0.0113).

The strongest recall is class 4 (0.9845). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[  205 13211  1829  1960   913]
 [    0   521    16    17     2]
 [    0    23  1369    43    13]
 [    0     3     9   150     0]
 [    0    16     7     2  1583]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.
