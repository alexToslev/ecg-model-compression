# Setup and Reproducibility Notes

This document records the expected development environment and the commands
used to reproduce the project workflows. It supports WP1 by making the setup
clear before the remaining implementation work continues.

## Environment

Use Python 3.10 or newer. A virtual environment is recommended so TensorFlow
and the scientific Python packages do not depend on system-level installs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The required packages are listed in `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

## Dataset Layout

The code expects the preprocessed MIT-BIH heartbeat CSV files here:

```text
data/processed/
├── mitbih_train.csv
└── mitbih_test.csv
```

Each row should contain 187 ECG signal values and one class label in the final
column. The repository also supports synthetic demo data through `--demo-data`
for fast smoke tests when the real dataset is unavailable.

## Reproducibility Defaults

Training commands expose `--seed`, and the current default seed is `42`. Dataset
splitting also uses this seed so repeated runs keep the same validation split
when the input CSV files are unchanged.

Default normalization is `none`. The supported modes are:

- `none`
- `standard`
- `per_sample`

When comparing final results, use the same normalization mode, seed, batch
size, and epoch count for each related experiment.

## Output Layout

Generated files should be written under `results/`. Canonical final evidence
should use the folders listed in `docs/work_packages.md`.

Temporary runs should use `tmp_*` folders or `/tmp/...` paths and should not be
used as final report evidence unless a later work package explicitly promotes
them.

## Quick Smoke Checks

Use demo data for fast checks:

```bash
python -m src --help
python -m src visualize --demo-data --output-dir /tmp/ecg_visualize_smoke
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_mlp_smoke
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_cnn_smoke
```

Use the full dataset for final experiments:

```bash
python -m src visualize --data-dir data/processed --output-dir results/dataset_visualizations
python -m src train-mlp --data-dir data/processed --epochs 20 --output-dir results/baseline_mlp
python -m src train --data-dir data/processed --epochs 20 --output-dir results/baseline_cnn
```

## Matplotlib Cache Note

In restricted environments, Matplotlib may warn that `~/.config/matplotlib` is
not writable and create a temporary cache under `/tmp`. This is harmless for
the project commands. To avoid the warning, set `MPLCONFIGDIR` to a writable
directory:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

## Result Integrity Rules

- Do not fake missing metrics.
- Keep useful 20-epoch CNN results until they are intentionally superseded.
- Keep final outputs under clear `results/` subdirectories.
- Keep temporary/debug outputs out of the final report.
- Commit each small work package separately with a `WPx.y:` prefix.
