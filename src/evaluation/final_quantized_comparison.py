"""Generate final float32-versus-INT8 comparison plots for the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_LABELS = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]


def parse_args() -> argparse.Namespace:
    """Parse final-result directories and hardware deployment annotations."""
    parser = argparse.ArgumentParser(
        description="Create final baseline-vs-INT8 comparison plots for the cap-4 ECG CNN."
    )
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/class_weight_sweep_correct_data/cap_4"))
    parser.add_argument("--int8-dir", type=Path, default=Path("results/final_candidate_cap_4"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_cap4_comparison"))
    parser.add_argument("--esp32-latency-ms", type=float, default=52.0)
    parser.add_argument("--tensor-arena-kib", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    """Build final comparison plots, CSV summary, and Markdown summary."""
    args = parse_args()
    output_dir = args.output_dir
    quantized_plots_dir = output_dir / "quantized_plots"
    comparison_plots_dir = output_dir / "comparison_plots"
    quantized_plots_dir.mkdir(parents=True, exist_ok=True)
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)

    # Inputs are already produced by earlier training and quantization steps; this
    # script only formats the final evidence used by the report and poster.
    baseline_metrics = load_json(args.baseline_dir / "metrics.json")
    baseline_report = load_json(args.baseline_dir / "classification_report.json")
    baseline_confusion = np.loadtxt(args.baseline_dir / "confusion_matrix.csv", delimiter=",", dtype=int)

    int8_metrics = load_json(args.int8_dir / "int8_metrics.json")
    int8_report = load_json(args.int8_dir / "int8_classification_report.json")
    int8_confusion = np.loadtxt(args.int8_dir / "int8_confusion_matrix.csv", delimiter=",", dtype=int)

    plot_class_metrics(int8_report, quantized_plots_dir / "int8_class_metrics.png", "INT8 Quantized Per-Class Metrics")
    plot_confusion_matrix(int8_confusion, quantized_plots_dir / "int8_confusion_matrix.png", "INT8 Quantized Confusion Matrix")
    plot_normalized_confusion_matrix(
        int8_confusion,
        quantized_plots_dir / "int8_confusion_matrix_normalized.png",
        "INT8 Quantized Normalized Confusion Matrix",
    )

    plot_accuracy_loss_comparison(
        baseline_metrics,
        int8_metrics,
        comparison_plots_dir / "baseline_vs_int8_accuracy_loss.png",
    )
    plot_macro_metrics_comparison(
        baseline_metrics,
        int8_metrics,
        baseline_report,
        int8_report,
        comparison_plots_dir / "baseline_vs_int8_macro_metrics.png",
    )
    plot_per_class_f1_comparison(
        baseline_report,
        int8_report,
        comparison_plots_dir / "baseline_vs_int8_per_class_f1.png",
    )
    plot_model_size_comparison(
        baseline_metrics,
        int8_metrics,
        comparison_plots_dir / "baseline_vs_int8_model_size.png",
    )
    plot_parameters_memory_latency(
        baseline_metrics,
        int8_metrics,
        args.tensor_arena_kib,
        args.esp32_latency_ms,
        comparison_plots_dir / "int8_deployment_parameters_memory_latency.png",
    )
    plot_confusion_side_by_side(
        baseline_confusion,
        int8_confusion,
        comparison_plots_dir / "baseline_vs_int8_confusion_matrices.png",
    )

    summary = build_summary_table(
        baseline_metrics=baseline_metrics,
        int8_metrics=int8_metrics,
        baseline_report=baseline_report,
        int8_report=int8_report,
        tensor_arena_kib=args.tensor_arena_kib,
        esp32_latency_ms=args.esp32_latency_ms,
    )
    summary.to_csv(output_dir / "baseline_vs_int8_summary.csv", index=False)
    write_markdown_summary(output_dir / "baseline_vs_int8_summary.md", summary, args)

    print(f"Wrote quantized plots to {quantized_plots_dir}")
    print(f"Wrote comparison plots to {comparison_plots_dir}")
    print(f"Wrote summary to {output_dir / 'baseline_vs_int8_summary.md'}")


def load_json(path: Path) -> dict:
    """Read a JSON artifact from a training or quantization run."""
    return json.loads(path.read_text(encoding="utf-8"))


def class_ids(report: dict) -> list[str]:
    """Return class id keys from a scikit-learn classification report."""
    return sorted([key for key in report if key.isdigit()], key=int)


def plot_class_metrics(report: dict, output_path: Path, title: str) -> None:
    """Plot precision, recall, and F1-score for each INT8 class."""
    ids = class_ids(report)
    labels = [CLASS_LABELS[int(class_id)] for class_id in ids]
    precision = [report[class_id]["precision"] for class_id in ids]
    recall = [report[class_id]["recall"] for class_id in ids]
    f1 = [report[class_id]["f1-score"] for class_id in ids]

    x = np.arange(len(ids))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, precision, width, label="Precision", color="#3b82f6")
    ax.bar(x, recall, width, label="Recall", color="#10b981")
    ax.bar(x + width, f1, width, label="F1-score", color="#f59e0b")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path, title: str) -> None:
    """Plot an integer-count confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    format_confusion_axis(ax, confusion, title, integer_values=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normalized_confusion_matrix(confusion: np.ndarray, output_path: Path, title: str) -> None:
    """Plot row-normalized confusion values for per-class recall inspection."""
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(normalized, cmap="Greens", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    format_confusion_axis(ax, normalized, title, integer_values=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_confusion_axis(ax, matrix: np.ndarray, title: str, integer_values: bool) -> None:
    """Apply labels and cell annotations to a confusion-matrix axis."""
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(CLASS_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            # Keep labels legible regardless of whether a cell is near the
            # top or bottom of the colormap range.
            value = f"{int(matrix[row, col])}" if integer_values else f"{matrix[row, col]:.2f}"
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, value, ha="center", va="center", color=color, fontsize=8)


def plot_accuracy_loss_comparison(baseline_metrics: dict, int8_metrics: dict, output_path: Path) -> None:
    """Compare final float32 and INT8 accuracy/loss values."""
    metric_names = ["Accuracy", "Loss"]
    baseline_values = [baseline_metrics["test_accuracy"], baseline_metrics["test_loss"]]
    int8_values = [int8_metrics["int8_accuracy"], int8_metrics["int8_loss"]]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars_a = ax.bar(x - width / 2, baseline_values, width, label="Baseline float32", color="#2563eb")
    bars_b = ax.bar(x + width / 2, int8_values, width, label="Quantized INT8", color="#ea580c")
    ax.set_title("Baseline vs INT8 Accuracy and Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax, list(bars_a) + list(bars_b), decimals=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_macro_metrics_comparison(
    baseline_metrics: dict,
    int8_metrics: dict,
    baseline_report: dict,
    int8_report: dict,
    output_path: Path,
) -> None:
    """Compare macro and weighted metrics for float32 and INT8 outputs."""
    metric_names = ["Macro precision", "Macro recall", "Macro F1", "Weighted F1"]
    baseline_values = [
        baseline_report["macro avg"]["precision"],
        baseline_report["macro avg"]["recall"],
        baseline_metrics["macro_f1"],
        baseline_metrics["weighted_f1"],
    ]
    int8_values = [
        int8_metrics["int8_macro_precision"],
        int8_metrics["int8_macro_recall"],
        int8_metrics["int8_macro_f1"],
        int8_report["weighted avg"]["f1-score"],
    ]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars_a = ax.bar(x - width / 2, baseline_values, width, label="Baseline float32", color="#2563eb")
    bars_b = ax.bar(x + width / 2, int8_values, width, label="Quantized INT8", color="#ea580c")
    ax.set_title("Baseline vs INT8 Macro and Weighted Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax, list(bars_a) + list(bars_b), decimals=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1_comparison(baseline_report: dict, int8_report: dict, output_path: Path) -> None:
    """Compare per-class F1 before and after quantization."""
    ids = class_ids(baseline_report)
    baseline_values = [baseline_report[class_id]["f1-score"] for class_id in ids]
    int8_values = [int8_report[class_id]["f1-score"] for class_id in ids]
    x = np.arange(len(ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars_a = ax.bar(x - width / 2, baseline_values, width, label="Baseline float32", color="#2563eb")
    bars_b = ax.bar(x + width / 2, int8_values, width, label="Quantized INT8", color="#ea580c")
    ax.set_title("Baseline vs INT8 Per-Class F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[int(class_id)] for class_id in ids])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("F1-score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax, list(bars_a) + list(bars_b), decimals=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_size_comparison(baseline_metrics: dict, int8_metrics: dict, output_path: Path) -> None:
    """Plot the storage reduction from Keras float32 to TFLite INT8."""
    labels = ["Baseline float32", "Quantized INT8"]
    values_kib = [
        baseline_metrics["keras_model_size_bytes"] / 1024.0,
        int8_metrics["tflite_size_bytes"] / 1024.0,
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(labels, values_kib, color=["#2563eb", "#ea580c"])
    ax.set_title("Baseline vs INT8 Model Size")
    ax.set_ylabel("Size (KiB)")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values_kib):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.1f} KiB", ha="center")
    reduction = int8_metrics["size_reduction_percent"]
    ax.text(0.5, max(values_kib) * 0.55, f"{reduction:.1f}% smaller", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_parameters_memory_latency(
    baseline_metrics: dict,
    int8_metrics: dict,
    tensor_arena_kib: float,
    esp32_latency_ms: float,
    output_path: Path,
) -> None:
    """Plot model parameters, memory footprint, and measured ESP32 latency."""
    labels = ["Trainable params", "Model size (KiB)", "Tensor arena (KiB)", "ESP32 latency (ms)"]
    baseline_values = [
        baseline_metrics["parameters"],
        baseline_metrics["keras_model_size_bytes"] / 1024.0,
        0.0,
        0.0,
    ]
    int8_values = [
        baseline_metrics["parameters"],
        int8_metrics["tflite_size_bytes"] / 1024.0,
        tensor_arena_kib,
        esp32_latency_ms,
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(x - width / 2, baseline_values, width, label="Baseline float32", color="#2563eb")
    bars_b = ax.bar(x + width / 2, int8_values, width, label="Quantized INT8 / ESP32", color="#ea580c")
    ax.set_title("Parameters, Memory Footprint, and ESP32 Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax, list(bars_a) + list(bars_b), decimals=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_side_by_side(baseline_confusion: np.ndarray, int8_confusion: np.ndarray, output_path: Path) -> None:
    """Place float32 and INT8 confusion matrices in one comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, matrix, title in [
        (axes[0], baseline_confusion, "Baseline Float32"),
        (axes[1], int8_confusion, "Quantized INT8"),
    ]:
        ax.imshow(matrix, cmap="Blues")
        format_confusion_axis(ax, matrix, title, integer_values=True)
    fig.suptitle("Baseline vs INT8 Confusion Matrices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def annotate_bars(ax, bars, decimals: int) -> None:
    """Add compact numeric labels above nonzero bars."""
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        label = f"{height:.{decimals}f}"
        ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom", fontsize=8)


def build_summary_table(
    baseline_metrics: dict,
    int8_metrics: dict,
    baseline_report: dict,
    int8_report: dict,
    tensor_arena_kib: float,
    esp32_latency_ms: float,
) -> pd.DataFrame:
    """Collect final float32, INT8, and hardware values into one table."""
    rows = [
        ("Test accuracy", baseline_metrics["test_accuracy"], int8_metrics["int8_accuracy"]),
        ("Test loss", baseline_metrics["test_loss"], int8_metrics["int8_loss"]),
        ("Macro precision", baseline_report["macro avg"]["precision"], int8_metrics["int8_macro_precision"]),
        ("Macro recall", baseline_report["macro avg"]["recall"], int8_metrics["int8_macro_recall"]),
        ("Macro F1", baseline_metrics["macro_f1"], int8_metrics["int8_macro_f1"]),
        ("Weighted F1", baseline_metrics["weighted_f1"], int8_report["weighted avg"]["f1-score"]),
        ("Trainable parameters", baseline_metrics["parameters"], baseline_metrics["parameters"]),
        ("Model size bytes", baseline_metrics["keras_model_size_bytes"], int8_metrics["tflite_size_bytes"]),
        ("Size reduction percent", 0.0, int8_metrics["size_reduction_percent"]),
        ("Tensor arena KiB", np.nan, tensor_arena_kib),
        ("ESP32 latency ms per beat", np.nan, esp32_latency_ms),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Baseline float32", "Quantized INT8"])


def write_markdown_summary(output_path: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write the final comparison summary used by the report deliverable."""
    lines = [
        "# Final Cap-4 Baseline vs INT8 Comparison",
        "",
        "## Summary table",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Generated quantized plots",
        "",
        "- `quantized_plots/int8_class_metrics.png`",
        "- `quantized_plots/int8_confusion_matrix.png`",
        "- `quantized_plots/int8_confusion_matrix_normalized.png`",
        "",
        "## Generated comparison plots",
        "",
        "- `comparison_plots/baseline_vs_int8_accuracy_loss.png`",
        "- `comparison_plots/baseline_vs_int8_macro_metrics.png`",
        "- `comparison_plots/baseline_vs_int8_per_class_f1.png`",
        "- `comparison_plots/baseline_vs_int8_model_size.png`",
        "- `comparison_plots/int8_deployment_parameters_memory_latency.png`",
        "- `comparison_plots/baseline_vs_int8_confusion_matrices.png`",
        "",
        "## Deployment notes",
        "",
        f"- ESP32 latency is recorded as approximately {args.esp32_latency_ms:.1f} ms per ECG beat from the hardware serial output.",
        f"- Tensor arena is recorded as {args.tensor_arena_kib:.1f} KiB from the deployed firmware configuration.",
        "- Full test-set accuracy is measured on PC; ESP32 execution validates that the quantized model runs on hardware with real representative MIT-BIH beats.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small pandas DataFrame as a plain Markdown table."""
    headers = list(df.columns)
    rows = [[format_markdown_value(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(headers[col])), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def format_markdown_value(value) -> str:
    """Format numbers and missing values for Markdown table cells."""
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):.0f}"
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
