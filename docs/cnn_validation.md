# CNN Baseline Validation

This document supports WP6.3. It records the commands used to validate the CNN
baseline workflow without overwriting the preserved 20-epoch results.

## Smoke Validation

Run a one-epoch demo-data training pass:

```bash
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_wp6_cnn_smoke
```

Expected behavior:

- the from-scratch CNN trains for one epoch
- dataset visualizations are generated
- evaluation artifacts are saved
- a Keras export is created for later TensorFlow Lite conversion

## Full-Data Prerequisite Check

Before running the full 20-epoch baseline, confirm the dataset files exist:

```bash
find data/processed -maxdepth 1 -type f -name 'mitbih_*.csv' -printf '%f %s\n'
```

The expected files are:

- `mitbih_train.csv`
- `mitbih_test.csv`

## Full Baseline Command

Use this command to regenerate the canonical baseline only when intended:

```bash
python -m src train --data-dir data/processed --epochs 20 --output-dir results/baseline_cnn --seed 42 --normalize none
```

This command is intentionally not used for quick validation because it can
overwrite the preserved 20-epoch baseline. For trial runs, use a temporary
output directory:

```bash
python -m src train --data-dir data/processed --epochs 20 --output-dir /tmp/ecg_cnn_full_trial --seed 42 --normalize none
```

## Baseline Artifact Check

The current preserved baseline should contain:

```bash
python -m json.tool results/baseline_cnn/metrics.json
python -m json.tool results/baseline_cnn/classification_report.json
sed -n '1,5p' results/baseline_cnn/history.csv
```

The current baseline inventory is recorded in `docs/cnn_baseline_results.md`.

## Notes

The CNN implementation uses scratch NumPy layers for training. Keras export is
only an interoperability step for evaluation and later TFLite quantization.
