# From-Scratch Improved CNN Summary

## What was run

A manual NumPy CNN was trained with explicit forward pass, backward pass, weighted softmax cross-entropy, Adam/SGD updates, and rare-class augmentation. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.1985
- Final validation accuracy: 0.2083
- Test accuracy: 0.2000
- Test loss: 1.5579
- Trainable parameters: 11173
- Saved Keras model: `results\smoke_scratch_cnn\tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=24, precision=0.0000, recall=0.0000, f1=0.0000
- Class 1: support=24, precision=0.0000, recall=0.0000, f1=0.0000
- Class 2: support=24, precision=0.0000, recall=0.0000, f1=0.0000
- Class 3: support=24, precision=0.2000, recall=1.0000, f1=0.3333
- Class 4: support=24, precision=0.0000, recall=0.0000, f1=0.0000

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 20.00%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 0 has the weakest recall (0.0000).

The strongest recall is class 3 (1.0000). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[ 0  0  0 24  0]
 [ 0  0  0 24  0]
 [ 0  0  0 24  0]
 [ 0  0  0 24  0]
 [ 0  0  0 24  0]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare macro F1 and rare-class recall, then tune class-weight cap and augmentation strength.
