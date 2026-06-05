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
    if train_csv.exists() and test_csv.exists():
        return

    raise FileNotFoundError(
        "MIT-BIH CSV files were not found.\n\n"
        f"Expected:\n  {train_csv}\n  {test_csv}\n\n"
        "Use the preprocessed MIT-BIH heartbeat CSV dataset with one heartbeat per row, "
        "187 signal values, and the class label in the final column. A common source is "
        "the Kaggle 'Heartbeat Categorization Dataset'."
    )


def load_mitbih_csv(
    data_dir: Path,
    validation_fraction: float = 0.15,
    normalize: str = "none",
    seed: int = 42,
) -> DatasetBundle:
    ensure_dataset_exists(data_dir)
    train_csv, test_csv = expected_csv_paths(data_dir)

    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    x_train_full, y_train_full = _split_features_and_labels(train_df)
    x_test, y_test = _split_features_and_labels(test_df)

    num_classes = int(max(y_train_full.max(), y_test.max())) + 1
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_full,
    )

    x_train, x_val, x_test = _normalize(x_train, x_val, x_test, normalize)

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
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    t = np.linspace(0.0, 1.0, input_length, dtype=np.float32)

    for label in range(num_classes):
        for _ in range(samples_per_class):
            center = 0.35 + 0.08 * label + rng.normal(0.0, 0.01)
            width = 0.025 + 0.004 * label
            r_peak = np.exp(-0.5 * ((t - center) / width) ** 2)
            p_wave = 0.18 * np.exp(-0.5 * ((t - (center - 0.18)) / 0.04) ** 2)
            twave = 0.28 * np.exp(-0.5 * ((t - (center + 0.22)) / 0.06) ** 2)
            baseline = 0.03 * np.sin(2 * np.pi * (2 + label) * t)
            noise = rng.normal(0.0, 0.025, size=input_length)
            signal = p_wave + r_peak + twave + baseline + noise
            xs.append(signal.astype(np.float32))
            ys.append(label)

    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_full,
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
    x = values[:, :-1]
    y = values[:, -1].astype(np.int64)
    return x, y


def _add_channel_axis(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)[..., np.newaxis]


def _normalize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "none":
        return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)
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


def _per_sample_standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-7
    return ((x - mean) / std).astype(np.float32)
