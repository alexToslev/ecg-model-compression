from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    input_length: int


def expected_csv_paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "mitbih_train.csv", data_dir / "mitbih_test.csv"


def ensure_dataset_exists(data_dir: Path) -> None:
    train_csv, test_csv = expected_csv_paths(data_dir)
    missing = [path for path in (train_csv, test_csv) if not path.exists()]
    if not missing:
        return

    raise FileNotFoundError(
        "MIT-BIH CSV files were not found.\n\n"
        f"Expected:\n  {train_csv}\n  {test_csv}\n\n"
        "Please place the preprocessed MIT-BIH heartbeat CSV dataset into the `data/processed/` folder. "
        "Each file should contain one heartbeat per row, 187 ECG values, and the class label in the final column."
    )


def load_mitbih_csv(
    data_dir: Path,
    validation_fraction: float = 0.15,
    normalize: str = "none",
    seed: int = 42,
) -> DatasetBundle:
    """Load MIT-BIH CSV files, normalize data, and split into train/val/test."""
    _validate_validation_fraction(validation_fraction)
    ensure_dataset_exists(data_dir)
    train_csv, test_csv = expected_csv_paths(data_dir)

    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    x_train_full, y_train_full = _split_features_and_labels(train_df)
    x_test, y_test = _split_features_and_labels(test_df)
    _validate_dataset_arrays(x_train_full, y_train_full, x_test, y_test)

    x_train, x_val, y_train, y_val = _split_train_val(
        x_train_full, y_train_full, validation_fraction, seed
    )

    x_train, x_val, x_test = _normalize(x_train, x_val, x_test, normalize)

    num_classes = int(max(np.max(y_train_full), np.max(y_test))) + 1

    return DatasetBundle(
        x_train=_add_channel_axis(x_train),
        y_train=y_train,
        x_val=_add_channel_axis(x_val),
        y_val=y_val,
        x_test=_add_channel_axis(x_test),
        y_test=y_test,
        num_classes=num_classes,
        input_length=x_train.shape[1],
    )


def make_demo_dataset(
    samples_per_class: int = 120,
    input_length: int = 187,
    num_classes: int = 5,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    """Generate a synthetic ECG-like dataset for development and quick testing."""
    _validate_validation_fraction(validation_fraction)
    if samples_per_class < 3:
        raise ValueError("samples_per_class must be at least 3 so train/validation/test splits are possible.")
    if input_length < 8:
        raise ValueError("input_length must be at least 8 for the CNN pooling stack.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, input_length, dtype=np.float32)

    x_all = []
    y_all = []
    for label in range(num_classes):
        for _ in range(samples_per_class):
            x_all.append(_simulate_ecg_waveform(t, label, rng))
            y_all.append(label)

    x = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_train, x_val, y_train, y_val = _split_train_val(
        x_train, y_train, validation_fraction, seed
    )

    x_train, x_val, x_test = _normalize(x_train, x_val, x_test, "standard")

    return DatasetBundle(
        x_train=_add_channel_axis(x_train),
        y_train=y_train,
        x_val=_add_channel_axis(x_val),
        y_val=y_val,
        x_test=_add_channel_axis(x_test),
        y_test=y_test,
        num_classes=num_classes,
        input_length=input_length,
    )


def _split_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = df.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("CSV input must contain at least one feature column and one label column.")
    x = values[:, :-1]
    raw_y = values[:, -1]
    if not np.all(np.isfinite(values)):
        raise ValueError("CSV input contains NaN or infinite values.")
    if not np.all(np.equal(raw_y, np.floor(raw_y))):
        raise ValueError("Class labels must be integer values in the final CSV column.")
    y = raw_y.astype(np.int64)
    if np.any(y < 0):
        raise ValueError("Class labels must be non-negative integer values.")
    return x, y


def _split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        raise ValueError("At least two classes are required for stratified train/validation splitting.")
    if np.any(counts < 2):
        rare = labels[counts < 2].tolist()
        raise ValueError(f"Each class needs at least two training samples for stratified splitting. Rare classes: {rare}")

    return train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y,
    )


def _add_channel_axis(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)[..., np.newaxis]


def _normalize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    if mode == "none":
        return x_train, x_val, x_test

    if mode == "standard":
        mean = x_train.mean()
        std = x_train.std() + 1e-7
        return (
            ((x_train - mean) / std).astype(np.float32),
            ((x_val - mean) / std).astype(np.float32),
            ((x_test - mean) / std).astype(np.float32),
        )

    if mode == "per_sample":
        return (
            _per_sample_standardize(x_train),
            _per_sample_standardize(x_val),
            _per_sample_standardize(x_test),
        )

    raise ValueError("normalize must be one of: none, standard, per_sample")


def _validate_validation_fraction(validation_fraction: float) -> None:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1.")


def _validate_dataset_arrays(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            "Train and test CSV files must have the same number of ECG feature columns. "
            f"Got train={x_train.shape[1]} and test={x_test.shape[1]}."
        )
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Feature and label arrays must contain the same number of rows.")
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("Train and test CSV files must not be empty.")
    train_classes = set(np.unique(y_train).tolist())
    test_classes = set(np.unique(y_test).tolist())
    missing_from_train = sorted(test_classes - train_classes)
    if missing_from_train:
        raise ValueError(f"Test set contains classes missing from training data: {missing_from_train}")


def _per_sample_standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-7
    return ((x - mean) / std).astype(np.float32)


def _simulate_ecg_waveform(t: np.ndarray, label: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single synthetic ECG-like waveform for a given label."""
    rhythm = 1.0 + 0.08 * label
    p_center = 0.28 + 0.02 * label
    qrs_center = 0.5 + 0.03 * label
    t_center = 0.72 + 0.02 * label

    p_wave = 0.12 * np.exp(-0.5 * ((t - p_center) / 0.03) ** 2)
    qrs = 1.0 * np.exp(-0.5 * ((t - qrs_center) / 0.02) ** 2)
    t_wave = 0.18 * np.exp(-0.5 * ((t - t_center) / 0.04) ** 2)
    baseline = 0.04 * np.sin(2 * np.pi * rhythm * t)
    noise = rng.normal(0.0, 0.02 + 0.01 * label, size=t.shape)

    waveform = p_wave + qrs + t_wave + baseline + noise
    return waveform.astype(np.float32)
