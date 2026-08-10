# ESP32 Final Cap-4 Hardware Run

Date: 2026-08-10

This run verifies that the final fully INT8 ECG CNN executes successfully on
the ESP32 with one embedded MIT-BIH test beat from each class.

| Sample | CSV row | True class | Predicted class | Top score | Latency (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0.984375 | 52.047 |
| 1 | 18122 | 1 | 1 | 0.968750 | 51.655 |
| 2 | 18674 | 2 | 2 | 0.964844 | 51.636 |
| 3 | 20122 | 3 | 3 | 0.843750 | 51.668 |
| 4 | 20285 | 4 | 4 | 0.996094 | 51.676 |

## Summary

- Correct predictions: 5/5
- Mean inference latency: 51.736 ms per beat
- Minimum inference latency: 51.636 ms
- Maximum inference latency: 52.047 ms
- Model size: 23,576 bytes
- Tensor arena: 80 KiB

This is a deployment sanity test, not an estimate of complete test-set
accuracy. The complete PC-side INT8 evaluation remains the authoritative model
quality result (95.12% accuracy and 0.8055 macro F1).

The final firmware expands the embedded hardware check to 25 reproducibly
sampled beats (five per class, NumPy seed 42). That result is recorded in
`hardware_run_25_samples.md`.
