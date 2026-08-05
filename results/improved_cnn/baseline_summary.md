# Improved 1D CNN Summary

## What was run

A class-weighted improved 1D CNN was trained on preprocessed MIT-BIH heartbeat segments. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.8919
- Final validation accuracy: 0.8971
- Test accuracy: 0.9091
- Test loss: 0.2875
- Trainable parameters: 11509
- Saved Keras model: `results\improved_cnn\tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=18118, precision=0.9914, recall=0.9054, f1=0.9464
- Class 1: support=556, precision=0.2712, recall=0.7896, f1=0.4037
- Class 2: support=1448, precision=0.8295, recall=0.9275, f1=0.8758
- Class 3: support=162, precision=0.3258, recall=0.8889, f1=0.4768
- Class 4: support=1608, precision=0.9441, recall=0.9776, f1=0.9606

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 90.91%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.7896).

The strongest recall is class 4 (0.9776). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[16404  1147   247   240    80]
 [   94   439     9    12     2]
 [   28    24  1343    42    11]
 [    6     0    12   144     0]
 [   15     9     8     4  1572]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Run int8 TensorFlow Lite quantization and compare accuracy, macro F1, minority-class recall, and model size against this float32 model.
