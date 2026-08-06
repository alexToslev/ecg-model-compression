# From-Scratch Improved CNN Summary

## What was run

A manual NumPy CNN was trained with explicit forward pass, backward pass, weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.9539
- Final validation accuracy: 0.9568
- Test accuracy: 0.9526
- Test loss: 0.1950
- Trainable parameters: 11173
- Saved Keras model: `results/sampling_sweep_cap_4/shuffle/tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=18118, precision=0.9830, recall=0.9644, f1=0.9736
- Class 1: support=556, precision=0.6810, recall=0.5683, f1=0.6196
- Class 2: support=1448, precision=0.7961, recall=0.9489, f1=0.8658
- Class 3: support=162, precision=0.4820, recall=0.8272, f1=0.6091
- Class 4: support=1608, precision=0.9454, recall=0.9689, f1=0.9570

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 95.26%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.5683).

The strongest recall is class 4 (0.9689). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[17473   144   315   110    76]
 [  210   316    16     7     7]
 [   39     1  1374    27     7]
 [   17     0    11   134     0]
 [   37     3    10     0  1558]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.
