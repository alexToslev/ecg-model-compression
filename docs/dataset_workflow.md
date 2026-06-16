# ECG Dataset Workflow

This document describes the WP2 dataset workflow: loading, validation,
normalization, train/validation/test splitting, visualization, and diagnostics.

## Expected Input

Place the preprocessed MIT-BIH heartbeat CSV files here:

```text
data/processed/
├── mitbih_train.csv
└── mitbih_test.csv
```

Each row must contain ECG signal values followed by the class label in the final
column. For the common Kaggle heartbeat CSV format, each row has 187 signal
values and one label.

The CSV files are intentionally ignored by Git because they are downloaded data,
not source code.

## Loader Checks

`src/data/mitbih_csv.py` validates the dataset before returning arrays:

- both expected CSV files exist
- train and test files are not empty
- train and test files have the same number of feature columns
- all values are finite
- labels are integer values
- labels are non-negative
- test classes also exist in the training data
- validation split fraction is between 0 and 1
- stratified splitting has enough samples per class

These checks make data errors fail early instead of producing misleading model
metrics later.

## Normalization Modes

The command-line tools support:

- `none`: use the CSV values as provided
- `standard`: subtract the train-set mean and divide by the train-set standard
  deviation
- `per_sample`: standardize each heartbeat independently

For final comparisons, keep the same normalization mode across related
experiments.

## Visualization and Diagnostics

Generate dataset plots and diagnostics with demo data:

```bash
python -m src visualize --demo-data --output-dir /tmp/ecg_visualize_demo
```

Generate them with the full dataset:

```bash
python -m src visualize --data-dir data/processed --output-dir results/dataset_visualizations
```

The visualization workflow writes:

- `train_signal_examples.png`
- `train_class_distribution.png`
- `dataset_split_distribution.png`
- `dataset_diagnostics.json`
- `dataset_diagnostics.md`

The diagnostics files record sample counts, split shapes, class counts, and
basic signal statistics. These files are useful for the final report because
they prove which data split and input shape were used.

## Smoke Commands

Use these commands after WP2 changes:

```bash
python -m src visualize --demo-data --output-dir /tmp/ecg_wp2_visualize
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_wp2_mlp
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_wp2_cnn
```

The demo commands should finish quickly and should not require the real
MIT-BIH CSV files.

## Final Dataset Command

Before final model training, regenerate canonical dataset diagnostics:

```bash
python -m src visualize --data-dir data/processed --output-dir results/dataset_visualizations --normalize none
```

Use the same `--normalize`, `--validation-fraction`, and `--seed` values that
will be used for final training.
