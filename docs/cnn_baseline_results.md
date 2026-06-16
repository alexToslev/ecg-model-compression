# Baseline CNN Result Inventory

This document supports WP6.1. It records the useful saved 20-epoch baseline CNN
run so later pruning, quantization, and benchmarking work can use it as the
target reference.

## Canonical Baseline CNN Folder

Use this folder as the current canonical baseline CNN evidence:

```text
results/baseline_cnn/
```

Observed baseline metrics:

| metric | value |
|---|---:|
| epochs | 20 |
| test accuracy | 0.875327 |
| test loss | 0.383516 |
| input length | 187 |
| classes | 5 |
| parameters | 49781 |
| NumPy weights size | 400352 bytes |
| Keras export size | 247144 bytes |

The training history has 20 rows in `results/baseline_cnn/history.csv`. The
last recorded validation accuracy is about `0.942928`, and the final training
accuracy is about `0.943614`.

## Files to Preserve

Preserve these as the useful baseline CNN artifacts unless a later clean rerun
intentionally supersedes them:

- `results/baseline_cnn/tiny_ecg_cnn_weights.npz`
- `results/baseline_cnn/tiny_ecg_cnn.keras`
- `results/baseline_cnn/metrics.json`
- `results/baseline_cnn/history.csv`
- `results/baseline_cnn/classification_report.json`
- `results/baseline_cnn/confusion_matrix.csv`
- `results/baseline_cnn/baseline_summary.md`
- `results/baseline_cnn/plots/learning_curves.png`
- `results/baseline_cnn/plots/class_metrics.png`
- `results/baseline_cnn/plots/confusion_matrix.png`
- `results/baseline_cnn/dataset_visualizations/`

Also keep the existing int8 export for WP8 comparison:

- `results/baseline_cnn/tiny_ecg_cnn_int8.tflite`
- `results/baseline_cnn/int8_metrics.json`

## Overlapping or Legacy Files

The folder also contains `baseline_ecg_cnn.keras`, which is much larger than
`tiny_ecg_cnn.keras`. It appears to be an older overlapping export. Do not
delete it during WP6. It can be reviewed during the final cleanup work package
after the canonical baseline and export path are confirmed.

`results/baseline_cnn/training_history.png` overlaps with
`results/baseline_cnn/plots/learning_curves.png`. Keep it for now; final cleanup
can decide whether the duplicate plot is still needed.

## Rule for Later Work

Do not overwrite `results/baseline_cnn/` during smoke tests. Use `/tmp/...` or a
new clearly named output directory. Only regenerate the canonical baseline with
an explicit full-run command and record the command, seed, normalization mode,
and epoch count.
