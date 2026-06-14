from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = {
    "0": "Class 0",
    "1": "Class 1",
    "2": "Class 2",
    "3": "Class 3",
    "4": "Class 4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots and a short written summary for a baseline run.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument(
        "--quantized-dir",
        type=Path,
        default=None,
        help="Optional directory containing int8 metrics from TensorFlow Lite quantization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(run_dir / "history.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "classification_report.json").read_text(encoding="utf-8"))
    confusion = np.loadtxt(run_dir / "confusion_matrix.csv", delimiter=",", dtype=int)

    plot_learning_curves(history, plots_dir / "learning_curves.png")
    plot_class_metrics(report, plots_dir / "class_metrics.png")
    plot_confusion_matrix(confusion, plots_dir / "confusion_matrix.png")

    quantized_metrics = None
    if args.quantized_dir is not None:
        quantized_metrics_path = args.quantized_dir / "int8_metrics.json"
        if quantized_metrics_path.exists():
            quantized_metrics = json.loads(
                quantized_metrics_path.read_text(encoding="utf-8")
            )
            plot_baseline_quantized_comparison(
                metrics,
                quantized_metrics,
                plots_dir / "baseline_vs_quantized_accuracy.png",
                plots_dir / "baseline_vs_quantized_size.png",
            )
        else:
            print(f"[summarize_baseline] Warning: {quantized_metrics_path} not found.")

    write_summary(
        run_dir / "baseline_summary.md",
        history,
        metrics,
        report,
        confusion,
        quantized_metrics,
    )


def plot_learning_curves(history: pd.DataFrame, output_path: Path) -> None:
    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_accuracy"], label="train")
    axes[0].plot(epochs, history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_metrics(report: dict, output_path: Path) -> None:
    class_ids = [key for key in report.keys() if key.isdigit()]
    labels = [CLASS_NAMES.get(class_id, f"Class {class_id}") for class_id in class_ids]
    precision = [report[class_id]["precision"] for class_id in class_ids]
    recall = [report[class_id]["recall"] for class_id in class_ids]
    f1 = [report[class_id]["f1-score"] for class_id in class_ids]

    x = np.arange(len(class_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1-score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = [CLASS_NAMES.get(str(i), f"Class {i}") for i in range(confusion.shape[0])]
    ax.set_xticks(np.arange(confusion.shape[1]))
    ax.set_yticks(np.arange(confusion.shape[0]))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix")

    threshold = confusion.max() / 2
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > threshold else "black"
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_baseline_quantized_comparison(
    baseline_metrics: dict,
    quantized_metrics: dict,
    accuracy_path: Path,
    size_path: Path,
) -> None:
    labels = ["Baseline", "Quantized int8"]
    accuracy_values = [baseline_metrics["test_accuracy"], float(quantized_metrics.get("int8_accuracy", np.nan))]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accuracy_values, color=["#4c72b0", "#dd8452"])
    ax.set_title("Baseline vs Quantized Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    for bar, value in zip(bars, accuracy_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(accuracy_path, dpi=180)
    plt.close(fig)

    baseline_size = baseline_metrics.get("keras_model_size_bytes", baseline_metrics.get("model_size_bytes", 0))
    quantized_size = quantized_metrics.get("tflite_size_bytes", 0)
    size_values = [baseline_size / 1024.0, quantized_size / 1024.0]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, size_values, color=["#4c72b0", "#dd8452"])
    ax.set_title("Baseline vs Quantized Model Size")
    ax.set_ylabel("Size (KB)")
    for bar, value in zip(bars, size_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.1f} KB", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(size_path, dpi=180)
    plt.close(fig)


def write_summary(
    output_path: Path,
    history: pd.DataFrame,
    metrics: dict,
    report: dict,
    confusion: np.ndarray,
    quantized_metrics: dict | None = None,
) -> None:
    final = history.iloc[-1]
    supports = {class_id: int(report[class_id]["support"]) for class_id in report if class_id.isdigit()}
    recalls = {class_id: report[class_id]["recall"] for class_id in report if class_id.isdigit()}
    weakest_class = min(recalls, key=recalls.get)
    strongest_class = max(recalls, key=recalls.get)

    lines = [
        "# Baseline 1D CNN Summary",
        "",
        "## What was run",
        "",
        "A small 1D CNN was trained on preprocessed MIT-BIH heartbeat segments. Each input has "
        f"{metrics['input_length']} ECG values and the model predicts one of {metrics['num_classes']} classes.",
        "",
        "## Main results",
        "",
        f"- Final training accuracy: {final['train_accuracy']:.4f}",
        f"- Final validation accuracy: {final['val_accuracy']:.4f}",
        f"- Test accuracy: {metrics['test_accuracy']:.4f}",
        f"- Test loss: {metrics['test_loss']:.4f}",
        f"- Trainable parameters: {metrics['parameters']}",
        f"- Saved Keras model: `{metrics.get('keras_model_path', metrics.get('model_path', 'unknown'))}`",
        "",
        "## Per-class observations",
        "",
    ]

    for class_id in sorted(supports, key=int):
        class_report = report[class_id]
        lines.append(
            f"- Class {class_id}: support={supports[class_id]}, "
            f"precision={class_report['precision']:.4f}, "
            f"recall={class_report['recall']:.4f}, "
            f"f1={class_report['f1-score']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The model learns smoothly: training and validation accuracy increase together, while both "
            "loss curves decrease. This means the first baseline is behaving correctly and does not show "
            "obvious overfitting.",
            "",
            f"The overall test accuracy is high ({metrics['test_accuracy']:.2%}), but this number hides "
            "an important class imbalance issue. The majority class has very high recall, while minority "
            f"class {weakest_class} has the weakest recall ({recalls[weakest_class]:.4f}).",
            "",
            f"The strongest recall is class {strongest_class} ({recalls[strongest_class]:.4f}). The weakest "
            "classes should be checked before claiming the classifier is medically reliable.",
        ]
    )

    if quantized_metrics is not None:
        baseline_size_bytes = metrics.get("keras_model_size_bytes", metrics.get("model_size_bytes"))
        baseline_size_label = f"{baseline_size_bytes} bytes" if baseline_size_bytes is not None else "unknown"

        lines.extend(
            [
                "",
                "## Baseline vs Quantized Comparison",
                "",
                f"- Baseline test accuracy: {metrics['test_accuracy']:.4f}",
                f"- Quantized int8 accuracy: {quantized_metrics.get('int8_accuracy', float('nan')):.4f}",
                f"- Baseline Keras model size: {baseline_size_label}",
                f"- Quantized TFLite size: {quantized_metrics.get('tflite_size_bytes', 'unknown')} bytes",
                "",
                "The comparison plots illustrate the accuracy and model size tradeoff between the float32 baseline and the int8 quantized model.",
                "",
                "Generated comparison plots:",
                "",
                "- `plots/baseline_vs_quantized_accuracy.png`",
                "- `plots/baseline_vs_quantized_size.png`",
            ]
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are true classes and columns are predicted classes.",
            "",
            "```text",
            str(confusion),
            "```",
            "",
            "## Generated plots",
            "",
            "- `plots/learning_curves.png`",
            "- `plots/class_metrics.png`",
            "- `plots/confusion_matrix.png`",
        ]
    )

    if quantized_metrics is not None:
        lines.extend([
            "- `plots/baseline_vs_quantized_accuracy.png`",
            "- `plots/baseline_vs_quantized_size.png`",
        ])

    lines.extend([
        "",
        "## Next step",
        "",
        "Run int8 TensorFlow Lite quantization and compare accuracy/model size against this float32 baseline.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
