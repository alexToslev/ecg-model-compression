# From-Scratch Improved CNN Summary

## What was run

A manual NumPy CNN was trained with explicit forward pass, backward pass, weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.8762
- Final validation accuracy: 0.8757
- Test accuracy: 0.8736
- Test loss: 0.4070
- Trainable parameters: 11173
- Saved Keras model: `results\improved_cnn_scratch\tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=18118, precision=0.9889, recall=0.8636, f1=0.9220
- Class 1: support=556, precision=0.2379, recall=0.7446, f1=0.3606
- Class 2: support=1448, precision=0.6863, recall=0.9517, f1=0.7975
- Class 3: support=162, precision=0.2058, recall=0.8704, f1=0.3329
- Class 4: support=1608, precision=0.9444, recall=0.9608, f1=0.9525

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 87.36%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.7446).

The strongest recall is class 4 (0.9608). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[15647  1306   578   500    87]
 [  111   414    16    15     0]
 [   29     8  1378    29     4]
 [    8     1    12   141     0]
 [   28    11    24     0  1545]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.
