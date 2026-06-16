from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, pruned, quantized, and QAT CNN results."
    )
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--pruning-dir", type=Path, default=Path("results/baseline_cnn/pruning"))
    parser.add_argument("--quantized-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--qat-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_cnn/quantization"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    baseline = _baseline_row(args.baseline_dir)
    rows.append(baseline)

    pruned = _pruned_row(args.pruning_dir, baseline)
    if pruned is not None:
        rows.append(pruned)

    quantized = _quantized_row(args.quantized_dir, baseline)
    if quantized is not None:
        rows.append(quantized)

    if args.qat_dir is not None:
        qat = _qat_row(args.qat_dir)
        if qat is not None:
            rows.append(qat)

    comparison = pd.DataFrame(rows)
    comparison_path = args.output_dir / "cnn_compression_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    write_summary(comparison, args.output_dir / "cnn_compression_comparison.md")
    plot_accuracy(comparison, args.output_dir / "cnn_compression_accuracy.png")
    plot_size(comparison, args.output_dir / "cnn_compression_size.png")
    plot_accuracy_size(comparison, args.output_dir / "cnn_compression_accuracy_vs_size.png")

    print(f"[compare_cnn] saved comparison to {comparison_path}")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _history_epochs(run_dir: Path, metrics: dict) -> int | None:
    if "epochs" in metrics:
        return int(metrics["epochs"])
    history_path = run_dir / "history.csv"
    if history_path.exists():
        return int(len(pd.read_csv(history_path)))
    return None


def _baseline_row(run_dir: Path) -> dict:
    metrics = _load_json(run_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(f"Missing baseline metrics: {run_dir / 'metrics.json'}")
    size_bytes = int(metrics.get("keras_model_size_bytes", metrics.get("model_size_bytes", 0)))
    return {
        "model": "baseline_cnn",
        "method": "float32",
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_loss": float(metrics["test_loss"]),
        "size_bytes": size_bytes,
        "parameters": int(metrics.get("parameters", 0)),
        "parameter_sparsity": 0.0,
        "structured_sparsity": 0.0,
        "epochs": _history_epochs(run_dir, metrics),
        "source": str(run_dir),
    }


def _pruned_row(pruning_dir: Path, baseline: dict) -> dict | None:
    metrics_path = pruning_dir / "pruning_metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    if "structured_sparsity" not in metrics.columns:
        metrics["structured_sparsity"] = np.nan

    candidates = metrics[metrics["prune_fraction"] > 0.0]
    if candidates.empty:
        candidates = metrics
    best = candidates.sort_values(["test_accuracy", "structured_sparsity"], ascending=[False, False]).iloc[0]
    return {
        "model": "structured_pruned_cnn",
        "method": "structured_pruning",
        "test_accuracy": float(best["test_accuracy"]),
        "test_loss": float(best["test_loss"]),
        "size_bytes": int(best["estimated_size_bytes"]),
        "parameters": int(best["nonzero_parameters"]),
        "parameter_sparsity": float(best["sparsity"]),
        "structured_sparsity": float(best.get("structured_sparsity", np.nan)),
        "epochs": baseline.get("epochs"),
        "source": str(metrics_path),
    }


def _quantized_row(quantized_dir: Path, baseline: dict) -> dict | None:
    metrics = _load_json(quantized_dir / "int8_metrics.json")
    if metrics is None:
        return None
    return {
        "model": "ptq_int8_cnn",
        "method": "post_training_quantization",
        "test_accuracy": float(metrics["int8_accuracy"]),
        "test_loss": float(metrics.get("int8_loss", np.nan)),
        "size_bytes": int(metrics["tflite_size_bytes"]),
        "parameters": baseline.get("parameters", 0),
        "parameter_sparsity": 0.0,
        "structured_sparsity": 0.0,
        "epochs": baseline.get("epochs"),
        "source": str(quantized_dir / "int8_metrics.json"),
    }


def _qat_row(qat_dir: Path) -> dict | None:
    metrics = _load_json(qat_dir / "metrics.json")
    if metrics is None:
        return None
    return {
        "model": "qat_cnn",
        "method": "quantization_aware_training",
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_loss": float(metrics["test_loss"]),
        "size_bytes": int(metrics.get("keras_model_size_bytes", metrics.get("model_size_bytes", 0))),
        "parameters": int(metrics.get("parameters", 0)),
        "parameter_sparsity": 0.0,
        "structured_sparsity": 0.0,
        "epochs": _history_epochs(qat_dir, metrics),
        "source": str(qat_dir),
    }


def write_summary(comparison: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# CNN Compression Comparison",
        "",
        "This report compares the baseline CNN with available structured-pruned, post-training quantized, and quantization-aware-trained CNN results.",
        "",
        "| model | method | test_accuracy | test_loss | size_bytes | parameters | parameter_sparsity | structured_sparsity | epochs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['method']} | {_format_float(row['test_accuracy'])} | "
            f"{_format_float(row['test_loss'])} | {int(row['size_bytes'])} | {int(row['parameters'])} | "
            f"{_format_float(row['parameter_sparsity'])} | {_format_float(row['structured_sparsity'])} | "
            f"{'' if pd.isna(row['epochs']) else int(row['epochs'])} |"
        )

    lines.extend([
        "",
        "Generated plots:",
        "",
        "- `cnn_compression_accuracy.png`",
        "- `cnn_compression_size.png`",
        "- `cnn_compression_accuracy_vs_size.png`",
        "",
        "Size note: pruned CNN size is estimated from nonzero float32 parameters unless the architecture is physically shrunk or sparse storage is used.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def plot_accuracy(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = comparison["model"].tolist()
    values = comparison["test_accuracy"].tolist()
    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868", "#dd8452", "#8172b3"][: len(labels)])
    ax.set_title("CNN Compression Accuracy Comparison")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_size(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = comparison["model"].tolist()
    values = (comparison["size_bytes"] / 1024.0).tolist()
    bars = ax.bar(labels, values, color=["#4c72b0", "#55a868", "#dd8452", "#8172b3"][: len(labels)])
    ax.set_title("CNN Compression Size Comparison")
    ax.set_ylabel("Size (KB)")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accuracy_size(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(comparison))
    labels = comparison["model"].tolist()
    accuracy = comparison["test_accuracy"].to_numpy()
    size_kb = (comparison["size_bytes"] / 1024.0).to_numpy()

    width = 0.38
    ax1.bar(x - width / 2, accuracy, width, color="#4c72b0", label="Accuracy")
    ax1.set_ylabel("Test accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, size_kb, width, color="#dd8452", label="Size")
    ax2.set_ylabel("Size (KB)")

    ax1.set_title("CNN Compression Accuracy and Size")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
