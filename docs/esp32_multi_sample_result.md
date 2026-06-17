# ESP32 Multi-Sample Inference Result

This note records the first multi-sample TensorFlow Lite Micro test on the ESP32-WROOM-32.

## Setup

- Board: ESP32-WROOM-32
- Arduino board selection: ESP32 Dev Module
- Model: full int8 TensorFlow Lite ECG CNN
- Test input: five hardcoded ECG heartbeat samples
- Selected samples: one sample from each expected class `0`, `1`, `2`, `3`, and `4`
- The ESP32 performs inference only. It does not train the model.

## Result Summary

The ESP32 ran inference successfully on all five samples.

```text
Correct samples: 2/5
Free heap after inference: 283480
```

This means the deployment works technically, but the model still struggles on some heartbeat classes.

## Serial Monitor Output

The following output was captured from the ESP32 Serial Monitor:

```text
  class 4: raw=-126 dequantized=0.007813
Sample 2 dataset_index=18674 expected=2 predicted=2 correct=yes time_us=99491
Output scores:
  class 0: raw=-124 dequantized=0.015625
  class 1: raw=-127 dequantized=0.003906
  class 2: raw=-14 dequantized=0.445312
  class 3: raw=-97 dequantized=0.121094
  class 4: raw=-23 dequantized=0.410156
Sample 3 dataset_index=20122 expected=3 predicted=0 correct=no time_us=99495
Output scores:
  class 0: raw=-44 dequantized=0.328125
  class 1: raw=-124 dequantized=0.015625
  class 2: raw=-51 dequantized=0.300781
  class 3: raw=-81 dequantized=0.183594
  class 4: raw=-84 dequantized=0.171875
Sample 4 dataset_index=20284 expected=4 predicted=0 correct=no time_us=99500
Output scores:
  class 0: raw=115 dequantized=0.949219
  class 1: raw=-127 dequantized=0.003906
  class 2: raw=-123 dequantized=0.019531
  class 3: raw=-124 dequantized=0.015625
  class 4: raw=-125 dequantized=0.011719
Correct samples: 2/5
Free heap after inference: 283480
```

## How To Read This

Each sample line contains:

- `dataset_index`: row index from the test CSV
- `expected`: true heartbeat class from the dataset
- `predicted`: class predicted by the ESP32 model
- `correct`: whether prediction matched the expected class
- `time_us`: inference time in microseconds

The inference time is about `99.5 ms` per heartbeat sample. The free heap after inference is still high, so memory is not the main problem for this small model.

## Class Meaning

| Label | Short name | Meaning |
| --- | --- | --- |
| 0 | N | Normal beat |
| 1 | S | Supraventricular ectopic beat |
| 2 | V | Ventricular ectopic beat |
| 3 | F | Fusion beat |
| 4 | Q | Unknown or unclassifiable beat |

## Interpretation

The result is useful even though the accuracy is low on this small five-sample test. It shows two different things:

1. ESP32 deployment is working: the int8 model loads, runs, and prints predictions.
2. The model is still weak for some non-normal heartbeat classes, especially classes that were already difficult in the earlier confusion matrix.

This matches the known class imbalance problem in the MIT-BIH heartbeat dataset. Overall test accuracy can look high because class `0` is very common, but minority classes such as `1`, `3`, and `4` can still perform poorly.

## Next Step

Before changing the ESP32 code further, the next useful step is to improve and compare the PC-side model quality:

- evaluate the current Keras CNN per class
- test class weighting or balanced sampling during training
- compare the tiny CNN with the teammate's larger CNN
- only then export the best candidate to ESP32 again

The ESP32 test should be kept as a deployment benchmark. Model improvement should happen on the PC first, then the improved int8 model can be exported and tested on ESP32.
