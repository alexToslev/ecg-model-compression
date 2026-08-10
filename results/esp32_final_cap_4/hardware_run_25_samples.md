# ESP32 Balanced 25-Sample Hardware Run

Date: 2026-08-10

Board port: COM3

This run evaluates the final fully INT8 cap-4 CNN on 25 embedded MIT-BIH test
beats selected reproducibly without replacement: five beats per class using
NumPy seed 42.

## Result

- Correct predictions: 21/25
- Balanced-subset accuracy: 84.00%
- Mean inference latency: 52.090 ms per beat
- Minimum inference latency: 52.015 ms
- Maximum inference latency: 52.659 ms
- Embedded model size: 23,576 bytes
- Tensor arena: 80 KiB

## Per-class hardware result

| True class | Correct | Tested | Recall on subset |
| ---: | ---: | ---: | ---: |
| 0 | 5 | 5 | 100% |
| 1 | 4 | 5 | 80% |
| 2 | 5 | 5 | 100% |
| 3 | 3 | 5 | 60% |
| 4 | 4 | 5 | 80% |

## Confusion matrix

Rows are true classes and columns are predicted classes.

| True / predicted | 0 | 1 | 2 | 3 | 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 0 | 0 | 0 | 0 |
| 1 | 1 | 4 | 0 | 0 | 0 |
| 2 | 0 | 0 | 5 | 0 | 0 |
| 3 | 2 | 0 | 0 | 3 | 0 |
| 4 | 1 | 0 | 0 | 0 | 4 |

The four errors were classes 1, 3, and 4 being predicted as class 0. This is
consistent with the full test-set evaluation, where minority-class separation
is more difficult than detection of the dominant normal class.

This balanced 25-beat run is stronger deployment evidence than the earlier
five-beat demonstration, but it is still not a replacement for the complete
PC-side INT8 evaluation. The authoritative complete test-set results remain
95.12% accuracy and 0.8055 macro F1.

The complete serial output is preserved in
`hardware_run_25_samples_raw.txt`.
