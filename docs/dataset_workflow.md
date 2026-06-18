# ECG Dataset Workflow

This document describes the dataset workflow for the CNN-improvements branch.

## Expected Input

Place the preprocessed MIT-BIH heartbeat CSV files here:

```text
data/processed/
  mitbih_train.csv
  mitbih_test.csv
```

Each row must contain 187 ECG signal values followed by the class label in the final column.

The CSV files are ignored by Git because they are downloaded data, not source code.

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

## Normalization Modes

The command-line tools support:

- `none`: use the CSV values as provided
- `standard`: subtract the train-set mean and divide by the train-set standard deviation
- `per_sample`: standardize each heartbeat independently

Keep the same normalization mode when comparing float32 and int8 models.

## Visualization

Generate dataset plots with demo data:

```bash
python -m src visualize --demo-data --output-dir results/demo_dataset_visualizations
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

## Smoke Commands

```bash
python -m src visualize --demo-data --output-dir results/smoke_visualize
python -m src train --demo-data --epochs 1 --output-dir results/smoke_improved_cnn
```
