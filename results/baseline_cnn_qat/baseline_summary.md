# Quantization-Aware CNN Summary

## What was run

A from-scratch 1D CNN was trained with fake quantization on preprocessed MIT-BIH heartbeat segments. Each input has 187 ECG values and the model predicts one of 5 classes.

## Main results

- Final training accuracy: 0.8262
- Final validation accuracy: 0.8266
- Test accuracy: 0.8455
- Test loss: 0.4754
- Trainable parameters: 49781
- Saved Keras model: `results/baseline_cnn_qat/tiny_ecg_cnn.keras`

## Per-class observations

- Class 0: support=17513, precision=0.8451, recall=0.9978, f1=0.9151
- Class 1: support=285, precision=0.0000, recall=0.0000, f1=0.0000
- Class 2: support=1126, precision=0.9568, recall=0.2558, f1=0.4036
- Class 3: support=5, precision=0.0000, recall=0.0000, f1=0.0000
- Class 4: support=2078, precision=0.0000, recall=0.0000, f1=0.0000

## Interpretation

The model learns smoothly: training and validation accuracy increase together, while both loss curves decrease. This means the first baseline is behaving correctly and does not show obvious overfitting.

The overall test accuracy is 84.55%. This single number should be read together with the per-class metrics, because class imbalance can hide weak minority-class performance. Class 1 has the weakest recall (0.0000).

The strongest recall is class 0 (0.9978). The weakest classes should be checked before claiming the classifier is medically reliable.

## Confusion matrix

Rows are true classes and columns are predicted classes.

```text
[[17474    11    12     0    16]
 [  285     0     0     0     0]
 [  837     1   288     0     0]
 [    5     0     0     0     0]
 [ 2077     0     1     0     0]]
```

## Generated plots

- `plots/learning_curves.png`
- `plots/class_metrics.png`
- `plots/confusion_matrix.png`

## Next step

Compare this QAT CNN against the float32 baseline, structured-pruned CNN, and post-training int8 CNN.
