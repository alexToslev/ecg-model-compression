"""Generate plots and small diagnostics reports for ECG datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_examples(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    class_names: dict[int, str] | None = None,
    samples_per_class: int = 3,
) -> None:
    """Plot representative signals for each class in the dataset."""
    x = x.squeeze(-1)
    num_classes = len(np.unique(y))
    # Keep a stable grid shape even when there is only one class or one sample
    # per class, because matplotlib changes the axes type in those cases.
    figure, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 3, num_classes * 2.5))

    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
    if samples_per_class == 1:
        axes = axes[:, np.newaxis]

    for class_id in range(num_classes):
        class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
        class_indices = np.where(y == class_id)[0][:samples_per_class]
        if class_indices.size == 0:
            continue

        for sample_index, idx in enumerate(class_indices):
            axes[class_id, sample_index].plot(x[idx], linewidth=1.2)
            axes[class_id, sample_index].set_title(f"{class_name} sample {sample_index + 1}")
            axes[class_id, sample_index].set_xlabel("Time step")
            axes[class_id, sample_index].set_ylabel("Amplitude")
            axes[class_id, sample_index].grid(alpha=0.35)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_class_distribution(
    y: np.ndarray,
    output_path: Path,
    class_names: dict[int, str] | None = None,
) -> None:
    """Create a histogram of class distribution for a dataset."""
    labels, counts = np.unique(y, return_counts=True)
    names = [class_names.get(int(label), f"Class {int(label)}") if class_names else f"Class {int(label)}" for label in labels]

    figure, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color="#4c72b0")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_split_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
    class_names: dict[int, str] | None = None,
) -> None:
    """Plot the number of samples per split for each class."""
    labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    train_counts = [np.sum(y_train == label) for label in labels]
    val_counts = [np.sum(y_val == label) for label in labels]
    test_counts = [np.sum(y_test == label) for label in labels]
    names = [class_names.get(int(label), f"Class {int(label)}") if class_names else f"Class {int(label)}" for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    figure, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width, train_counts, width, label="train")
    ax.bar(x, val_counts, width, label="validation")
    ax.bar(x + width, test_counts, width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Dataset split distribution by class")
    ax.set_ylabel("Number of samples")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def visualize_dataset(
    dataset,
    output_dir: Path,
    class_names: dict[int, str] | None = None,
    samples_per_class: int = 3,
) -> None:
    """Generate key dataset visualizations for training and validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # These three plots are the quick sanity checks used throughout the report:
    # signal shape examples, class imbalance, and split consistency.
    plot_signal_examples(
        dataset.x_train,
        dataset.y_train,
        output_dir / "train_signal_examples.png",
        class_names,
        samples_per_class,
    )
    plot_class_distribution(
        dataset.y_train,
        output_dir / "train_class_distribution.png",
        class_names,
    )
    plot_split_distribution(
        dataset.y_train,
        dataset.y_val,
        dataset.y_test,
        output_dir / "dataset_split_distribution.png",
        class_names,
    )
    diagnostics = build_dataset_diagnostics(dataset)
    (output_dir / "dataset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    write_dataset_diagnostics_markdown(
        diagnostics,
        output_dir / "dataset_diagnostics.md",
    )


def build_dataset_diagnostics(dataset) -> dict:
    """Create a compact diagnostics dictionary for the loaded dataset."""
    split_arrays = {
        "train": (dataset.x_train, dataset.y_train),
        "validation": (dataset.x_val, dataset.y_val),
        "test": (dataset.x_test, dataset.y_test),
    }

    diagnostics = {
        "input_length": int(dataset.input_length),
        "num_classes": int(dataset.num_classes),
        "splits": {},
        "total_samples": int(len(dataset.y_train) + len(dataset.y_val) + len(dataset.y_test)),
    }

    for split_name, (x_split, y_split) in split_arrays.items():
        flattened = x_split.squeeze(-1)
        # Store plain Python numbers so the diagnostics JSON is easy to inspect
        # and does not depend on NumPy-specific serialization.
        diagnostics["splits"][split_name] = {
            "samples": int(len(y_split)),
            "shape": [int(value) for value in x_split.shape],
            "class_counts": _class_counts(y_split, dataset.num_classes),
            "signal_mean": float(np.mean(flattened)),
            "signal_std": float(np.std(flattened)),
            "signal_min": float(np.min(flattened)),
            "signal_max": float(np.max(flattened)),
        }

    return diagnostics


def write_dataset_diagnostics_markdown(diagnostics: dict, output_path: Path) -> None:
    """Write diagnostics as a small Markdown report for the project notebook/report."""
    lines = [
        "# Dataset Diagnostics",
        "",
        f"- Input length: {diagnostics['input_length']}",
        f"- Number of classes: {diagnostics['num_classes']}",
        f"- Total samples: {diagnostics['total_samples']}",
        "",
        "## Split Summary",
        "",
        "| split | samples | shape | mean | std | min | max |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]

    for split_name, split_info in diagnostics["splits"].items():
        lines.append(
            f"| {split_name} | {split_info['samples']} | {split_info['shape']} | "
            f"{split_info['signal_mean']:.6f} | {split_info['signal_std']:.6f} | "
            f"{split_info['signal_min']:.6f} | {split_info['signal_max']:.6f} |"
        )

    lines.extend([
        "",
        "## Class Counts",
        "",
        "| split | class | samples |",
        "|---|---:|---:|",
    ])

    for split_name, split_info in diagnostics["splits"].items():
        for class_id, count in split_info["class_counts"].items():
            lines.append(f"| {split_name} | {class_id} | {count} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _class_counts(y: np.ndarray, num_classes: int) -> dict[str, int]:
    """Return counts for every class, including classes absent from a split."""
    labels, counts = np.unique(y, return_counts=True)
    observed = {int(label): int(count) for label, count in zip(labels, counts)}
    return {str(class_id): observed.get(class_id, 0) for class_id in range(num_classes)}
