from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create presentation-ready plots from saved ECG experiment results.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--quantized-dir", type=Path, default=Path("results/quantized"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("results/scratch_cnn"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/presentation_plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = read_json(args.baseline_dir / "metrics.json")
    quant_metrics = read_json(args.quantized_dir / "quantization_metrics.json")
    scratch_metrics = read_json(args.scratch_dir / "metrics.json")

    plot_keras_training_curves(args.baseline_dir / "history.csv", args.output_dir / "keras_training_curves.png")
    plot_scratch_training_curves(args.scratch_dir / "history.csv", args.output_dir / "scratch_training_curves.png")
    plot_quantization_accuracy_and_size(quant_metrics, args.output_dir / "quantization_accuracy_size.png")
    plot_confusion_matrix(
        args.quantized_dir / "int8_confusion_matrix.csv",
        args.output_dir / "int8_confusion_matrix.png",
        "Int8 TFLite Confusion Matrix",
    )
    plot_confusion_matrix(
        args.scratch_dir / "confusion_matrix.csv",
        args.output_dir / "scratch_confusion_matrix.png",
        "From-Scratch CNN Confusion Matrix",
    )
    plot_accuracy_comparison(
        baseline_metrics,
        quant_metrics,
        scratch_metrics,
        args.output_dir / "model_accuracy_comparison.png",
    )
    plot_parameter_comparison(
        baseline_metrics,
        scratch_metrics,
        args.output_dir / "model_parameter_comparison.png",
    )

    print(f"Wrote presentation plots to {args.output_dir}")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_keras_training_curves(history_path: Path, output_path: Path) -> None:
    history = pd.read_csv(history_path)
    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["accuracy"], marker="o", label="train")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="validation")
    axes[0].set_title("Keras CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], marker="o", label="train")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[1].set_title("Keras CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    save_figure(fig, output_path)


def plot_scratch_training_curves(history_path: Path, output_path: Path) -> None:
    history = pd.read_csv(history_path)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="validation")
    axes[0].set_title("From-Scratch CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], marker="o")
    axes[1].set_title("From-Scratch CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    save_figure(fig, output_path)


def plot_quantization_accuracy_and_size(metrics: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    accuracy_labels = ["Float32 Keras", "Int8 TFLite"]
    accuracies = [metrics["float32_accuracy"], metrics["int8_accuracy"]]
    axes[0].bar(accuracy_labels, accuracies, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Accuracy Before/After Quantization")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(max(0.0, min(accuracies) - 0.02), 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(accuracies):
        axes[0].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    size_labels = ["Keras", "TFLite"]
    sizes_kb = [metrics["keras_size_bytes"] / 1024, metrics["tflite_size_bytes"] / 1024]
    axes[1].bar(size_labels, sizes_kb, color=["#4C78A8", "#F58518"])
    axes[1].set_title("Model File Size")
    axes[1].set_ylabel("Size (KB)")
    axes[1].grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(sizes_kb):
        axes[1].text(index, value, f"{value:.1f} KB", ha="center", va="bottom")

    save_figure(fig, output_path)


def plot_confusion_matrix(matrix_path: Path, output_path: Path, title: str) -> None:
    matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = CLASS_NAMES[: matrix.shape[0]]
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)

    save_figure(fig, output_path)


def plot_accuracy_comparison(
    baseline_metrics: dict,
    quant_metrics: dict,
    scratch_metrics: dict,
    output_path: Path,
) -> None:
    labels = ["Keras float32", "Keras int8", "From scratch"]
    accuracies = [
        baseline_metrics["test_accuracy"],
        quant_metrics["int8_accuracy"],
        scratch_metrics["test_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, accuracies, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(accuracies):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom")

    save_figure(fig, output_path)


def plot_parameter_comparison(
    baseline_metrics: dict,
    scratch_metrics: dict,
    output_path: Path,
) -> None:
    labels = ["Keras CNN", "From-scratch CNN"]
    parameters = [baseline_metrics["parameters"], scratch_metrics["parameters"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, parameters, color=["#4C78A8", "#54A24B"])
    ax.set_title("Parameter Count Comparison")
    ax.set_ylabel("Parameters")
    ax.grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(parameters):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")

    save_figure(fig, output_path)


def save_figure(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
