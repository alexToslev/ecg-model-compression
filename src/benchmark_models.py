from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d import build_tiny_cnn
from src.models.mlp import build_manual_mlp
from src.quantize_mlp import QuantizedDense, QuantizedManualMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark all available ECG compression models in one comparison table."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--baseline-mlp-dir", type=Path, default=Path("results/baseline_mlp"))
    parser.add_argument("--pruned-mlp-dir", type=Path, default=Path("results/baseline_mlp/pruning_structured"))
    parser.add_argument("--quantized-mlp-dir", type=Path, default=Path("results/baseline_mlp/quantization"))
    parser.add_argument("--baseline-cnn-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--pruned-cnn-dir", type=Path, default=Path("results/baseline_cnn/pruning"))
    parser.add_argument("--quantized-cnn-dir", type=Path, default=Path("results/baseline_cnn"))
    parser.add_argument("--qat-cnn-dir", type=Path, default=Path("results/baseline_cnn_qat"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timing-samples", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--demo-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = (
        make_demo_dataset()
        if args.demo_data
        else load_mitbih_csv(data_dir=args.data_dir, normalize=args.normalize)
    )
    x_timing = dataset.x_test[: min(args.timing_samples, len(dataset.x_test))]

    rows = [
        benchmark_baseline_mlp(args, dataset, x_timing),
        benchmark_pruned_mlp(args, dataset, x_timing),
        benchmark_quantized_mlp(args, dataset, x_timing),
        benchmark_baseline_cnn(args, dataset, x_timing),
        benchmark_pruned_cnn(args, dataset, x_timing),
        benchmark_quantized_cnn(args, dataset, x_timing),
        benchmark_qat_cnn(args, dataset, x_timing),
    ]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "model_benchmark_comparison.csv", index=False)
    write_markdown_summary(comparison, args, dataset)
    plot_benchmark_results(comparison, args.output_dir)
    print(f"[benchmark_models] saved benchmark comparison to {args.output_dir}")


def benchmark_baseline_mlp(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.baseline_mlp_dir
    metrics = load_json(run_dir / "metrics.json")
    weights_path = run_dir / "baseline_mlp_weights.npz"
    timing = np.nan
    if weights_path.exists():
        model = build_manual_mlp(dataset.input_length, dataset.num_classes)
        model.set_parameters(load_npz(weights_path))
        timing = time_batched_forward(lambda x: model.forward(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="baseline_mlp",
        family="MLP",
        method="float32",
        dataset_name=dataset_name(args),
        training_mode=metrics.get("training_mode", "float32"),
        epochs=history_epochs(run_dir, metrics),
        test_accuracy=metrics.get("test_accuracy"),
        test_loss=metrics.get("test_loss"),
        parameters=metrics.get("parameters"),
        parameter_sparsity=0.0,
        structured_sparsity=0.0,
        size_bytes=model_size(run_dir, metrics, ["keras_model_size_bytes", "weights_size_bytes"]),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="baseline scratch MLP",
    )


def benchmark_pruned_mlp(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.pruned_mlp_dir
    pruning = best_pruning_row(run_dir / "pruning_metrics.csv")
    timing = np.nan
    active_parameters = None
    estimated_size = None
    weights_path = pruning_weight_path(run_dir, pruning)
    if pruning is not None and weights_path.exists():
        model = build_manual_mlp(dataset.input_length, dataset.num_classes)
        model.set_parameters(load_npz(weights_path))
        active_parameters = int(round(model.parameter_count() * (1.0 - float(pruning.get("sparsity", 0.0)))))
        estimated_size = int(active_parameters * 4)
        timing = time_batched_forward(lambda x: model.forward(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="pruned_mlp",
        family="MLP",
        method=str(pruning.get("method", "structured_pruning")) if pruning is not None else "structured_pruning",
        dataset_name=dataset_name(args),
        training_mode="float32 + pruning",
        epochs=history_epochs(args.baseline_mlp_dir, load_json(args.baseline_mlp_dir / "metrics.json")),
        test_accuracy=None if pruning is None else pruning.get("test_accuracy"),
        test_loss=None if pruning is None else pruning.get("test_loss"),
        parameters=None if pruning is None else pruning.get("nonzero_parameters", active_parameters),
        parameter_sparsity=None if pruning is None else pruning.get("sparsity"),
        structured_sparsity=None if pruning is None else pruning.get("structured_sparsity"),
        size_bytes=None if pruning is None else pruning.get("estimated_size_bytes", estimated_size),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="best nonzero structured MLP pruning point by test accuracy",
    )


def benchmark_quantized_mlp(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.quantized_mlp_dir
    metrics = load_json(run_dir / "quantization_metrics.json")
    baseline_metrics = load_json(args.baseline_mlp_dir / "metrics.json")
    quantized_path = run_dir / "baseline_mlp_quantized.npz"
    timing = np.nan
    if quantized_path.exists():
        q_model = load_quantized_mlp(quantized_path)
        timing = time_batched_forward(lambda x: q_model.forward(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="quantized_mlp",
        family="MLP",
        method="manual_int8",
        dataset_name=dataset_name(args),
        training_mode="post_training_quantization",
        epochs=history_epochs(args.baseline_mlp_dir, load_json(args.baseline_mlp_dir / "metrics.json")),
        test_accuracy=metrics.get("quantized_test_accuracy"),
        test_loss=metrics.get("quantized_test_loss"),
        parameters=metrics.get("parameters", baseline_metrics.get("parameters")),
        parameter_sparsity=0.0,
        structured_sparsity=0.0,
        size_bytes=metrics.get("quantized_model_size_bytes"),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="manual fixed-point int8 MLP",
    )


def benchmark_baseline_cnn(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.baseline_cnn_dir
    metrics = load_json(run_dir / "metrics.json")
    weights_path = run_dir / "tiny_ecg_cnn_weights.npz"
    timing = np.nan
    if weights_path.exists():
        model = build_tiny_cnn(dataset.input_length, dataset.num_classes)
        model.set_parameters(load_npz(weights_path))
        timing = time_batched_forward(lambda x: model.forward(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="baseline_cnn",
        family="CNN",
        method="float32",
        dataset_name=dataset_name(args),
        training_mode=metrics.get("training_mode", "float32"),
        epochs=history_epochs(run_dir, metrics),
        test_accuracy=metrics.get("test_accuracy"),
        test_loss=metrics.get("test_loss"),
        parameters=metrics.get("parameters"),
        parameter_sparsity=0.0,
        structured_sparsity=0.0,
        size_bytes=model_size(run_dir, metrics, ["keras_model_size_bytes", "model_size_bytes"]),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="baseline scratch 1D CNN",
    )


def benchmark_pruned_cnn(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.pruned_cnn_dir
    pruning = best_pruning_row(run_dir / "pruning_metrics.csv")
    timing = np.nan
    weights_path = pruning_weight_path(run_dir, pruning)
    if pruning is not None and weights_path.exists():
        model = build_tiny_cnn(dataset.input_length, dataset.num_classes)
        model.set_parameters(load_npz(weights_path))
        timing = time_batched_forward(lambda x: model.forward(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="structured_pruned_cnn",
        family="CNN",
        method=str(pruning.get("method", "structured_pruning")) if pruning is not None else "structured_pruning",
        dataset_name=dataset_name(args),
        training_mode="float32 + structured pruning",
        epochs=history_epochs(args.baseline_cnn_dir, load_json(args.baseline_cnn_dir / "metrics.json")),
        test_accuracy=None if pruning is None else pruning.get("test_accuracy"),
        test_loss=None if pruning is None else pruning.get("test_loss"),
        parameters=None if pruning is None else pruning.get("nonzero_parameters", pruning.get("total_parameters")),
        parameter_sparsity=None if pruning is None else pruning.get("sparsity"),
        structured_sparsity=None if pruning is None else pruning.get("structured_sparsity"),
        size_bytes=None if pruning is None else pruning.get("estimated_size_bytes"),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="best nonzero structured CNN pruning point by test accuracy; size is estimated",
    )


def benchmark_quantized_cnn(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.quantized_cnn_dir
    metrics = load_json(run_dir / "int8_metrics.json")
    baseline = load_json(args.baseline_cnn_dir / "metrics.json")
    tflite_path = Path(metrics.get("tflite_path", run_dir / "tiny_ecg_cnn_int8.tflite"))
    if not tflite_path.is_absolute():
        tflite_path = Path(tflite_path)
    timing = time_tflite_model(tflite_path, x_timing, args.repeats, args.warmup) if tflite_path.exists() else np.nan

    return make_row(
        model_name="ptq_int8_cnn",
        family="CNN",
        method="post_training_int8_tflite",
        dataset_name=dataset_name(args),
        training_mode="post_training_quantization",
        epochs=history_epochs(args.baseline_cnn_dir, baseline),
        test_accuracy=metrics.get("int8_accuracy"),
        test_loss=metrics.get("int8_loss"),
        parameters=baseline.get("parameters"),
        parameter_sparsity=0.0,
        structured_sparsity=0.0,
        size_bytes=metrics.get("tflite_size_bytes"),
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="TensorFlow Lite int8 CNN",
    )


def benchmark_qat_cnn(args: argparse.Namespace, dataset, x_timing: np.ndarray) -> dict:
    run_dir = args.qat_cnn_dir
    metrics = load_json(run_dir / "metrics.json")
    weights_path = run_dir / "tiny_ecg_cnn_weights.npz"
    timing = np.nan
    if metrics and weights_path.exists():
        model = build_tiny_cnn(dataset.input_length, dataset.num_classes)
        model.set_parameters(load_npz(weights_path))
        timing = time_batched_forward(lambda x: model.forward_quantized(x), x_timing, args.batch_size, args.repeats, args.warmup)

    return make_row(
        model_name="qat_cnn",
        family="CNN",
        method="quantization_aware_training",
        dataset_name=dataset_name(args),
        training_mode=metrics.get("training_mode", "missing") if metrics else "missing",
        epochs=history_epochs(run_dir, metrics),
        test_accuracy=metrics.get("test_accuracy") if metrics else None,
        test_loss=metrics.get("test_loss") if metrics else None,
        parameters=metrics.get("parameters") if metrics else None,
        parameter_sparsity=0.0 if metrics else None,
        structured_sparsity=0.0 if metrics else None,
        size_bytes=model_size(run_dir, metrics, ["keras_model_size_bytes", "model_size_bytes"]) if metrics else None,
        inference_ms_per_sample=timing,
        timing_samples=len(x_timing),
        artifact_dir=run_dir,
        notes="missing; run WP8 QAT command" if not metrics else "fake-quantized scratch CNN",
        result_available=bool(metrics),
    )


def make_row(
    model_name: str,
    family: str,
    method: str,
    dataset_name: str,
    training_mode: str,
    epochs,
    test_accuracy,
    test_loss,
    parameters,
    parameter_sparsity,
    structured_sparsity,
    size_bytes,
    inference_ms_per_sample,
    timing_samples: int,
    artifact_dir: Path,
    notes: str,
    result_available: bool = True,
) -> dict:
    return {
        "model": model_name,
        "family": family,
        "method": method,
        "dataset": dataset_name,
        "training_mode": training_mode,
        "epochs": as_number(epochs),
        "test_accuracy": as_number(test_accuracy),
        "test_loss": as_number(test_loss),
        "parameters": as_number(parameters),
        "parameter_sparsity": as_number(parameter_sparsity),
        "structured_sparsity": as_number(structured_sparsity),
        "size_bytes": as_number(size_bytes),
        "inference_ms_per_sample": as_number(inference_ms_per_sample),
        "timing_samples": timing_samples,
        "artifact_dir": str(artifact_dir),
        "result_available": result_available,
        "notes": notes,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_quantized_mlp(path: Path) -> QuantizedManualMLP:
    arrays = load_npz(path)
    weight_keys = sorted([key for key in arrays if key.startswith("weights_")], key=lambda item: int(item.split("_")[1]))
    weight_scales = arrays["weight_scales"]
    bias_scales = arrays["bias_scales"]
    activation_scales = arrays["activation_scales"].astype(np.float32).tolist()

    dense_layers = []
    input_scale = float(arrays["input_scale"][0])
    input_scale_for_layer = input_scale
    for index, key in enumerate(weight_keys):
        output_scale = input_scale_for_layer * float(weight_scales[index])
        dense_layers.append(
            QuantizedDense(
                weights_int8=arrays[key],
                weight_scale=float(weight_scales[index]),
                bias_int32=arrays[f"bias_{index}"],
                output_scale=output_scale,
                bias_scale=float(bias_scales[index]),
            )
        )
        if index < len(activation_scales):
            input_scale_for_layer = activation_scales[index]

    original_size_bytes = int(sum(arrays[key].size * 4 for key in weight_keys))
    quantized_size_bytes = int(path.stat().st_size)
    return QuantizedManualMLP(
        dense_layers=dense_layers,
        input_scale=input_scale,
        activation_scales=activation_scales,
        original_size_bytes=original_size_bytes,
        quantized_size_bytes=quantized_size_bytes,
    )


def best_pruning_row(metrics_path: Path) -> pd.Series | None:
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    if "structured_sparsity" not in metrics.columns:
        metrics["structured_sparsity"] = np.nan
    if "estimated_size_bytes" not in metrics.columns and "nonzero_parameters" in metrics.columns:
        metrics["estimated_size_bytes"] = metrics["nonzero_parameters"] * 4
    if "nonzero_parameters" not in metrics.columns and "total_parameters" in metrics.columns and "sparsity" in metrics.columns:
        metrics["nonzero_parameters"] = (metrics["total_parameters"] * (1.0 - metrics["sparsity"])).round()
    candidates = metrics[metrics["prune_fraction"] > 0.0]
    if candidates.empty:
        candidates = metrics
    return candidates.sort_values(["test_accuracy", "sparsity"], ascending=[False, False]).iloc[0]


def pruning_weight_path(run_dir: Path, row: pd.Series | None) -> Path:
    if row is None:
        return run_dir / "missing.npz"
    fraction = int(round(float(row["prune_fraction"]) * 100))
    return run_dir / f"pruned_{fraction:02d}.npz"


def time_batched_forward(
    forward: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        for start in range(0, len(x), batch_size):
            forward(x[start : start + batch_size])

    start_time = time.perf_counter()
    for _ in range(repeats):
        for start in range(0, len(x), batch_size):
            forward(x[start : start + batch_size])
    elapsed = time.perf_counter() - start_time
    return float((elapsed / (len(x) * repeats)) * 1000.0)


def time_tflite_model(model_path: Path, x: np.ndarray, repeats: int, warmup: int) -> float:
    try:
        import tensorflow as tf
    except ImportError:
        return float("nan")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_scale, input_zero_point = input_details["quantization"]

    def invoke(sample: np.ndarray) -> None:
        if input_scale:
            sample = sample / input_scale + input_zero_point
            sample = np.clip(sample, -128, 127).astype(np.int8)
        else:
            sample = sample.astype(np.float32)
        interpreter.set_tensor(input_details["index"], sample[np.newaxis, ...])
        interpreter.invoke()

    for _ in range(warmup):
        for sample in x:
            invoke(sample)

    start_time = time.perf_counter()
    for _ in range(repeats):
        for sample in x:
            invoke(sample)
    elapsed = time.perf_counter() - start_time
    return float((elapsed / (len(x) * repeats)) * 1000.0)


def model_size(run_dir: Path, metrics: dict, keys: list[str]) -> float | None:
    for key in keys:
        if key in metrics:
            return metrics[key]
    for candidate in [
        run_dir / "baseline_mlp_weights.npz",
        run_dir / "tiny_ecg_cnn_weights.npz",
        run_dir / "baseline_mlp.keras",
        run_dir / "tiny_ecg_cnn.keras",
    ]:
        if candidate.exists():
            return candidate.stat().st_size
    return None


def history_epochs(run_dir: Path, metrics: dict) -> float | None:
    if metrics and "epochs" in metrics:
        return metrics["epochs"]
    history_path = run_dir / "history.csv"
    if history_path.exists():
        return len(pd.read_csv(history_path))
    return None


def dataset_name(args: argparse.Namespace) -> str:
    return "synthetic_demo" if args.demo_data else "MIT-BIH processed CSV"


def as_number(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def write_markdown_summary(comparison: pd.DataFrame, args: argparse.Namespace, dataset) -> None:
    available = comparison[comparison["result_available"] == True].copy()
    best_accuracy = available.loc[available["test_accuracy"].idxmax()] if available["test_accuracy"].notna().any() else None
    smallest = available.loc[available["size_bytes"].idxmin()] if available["size_bytes"].notna().any() else None
    fastest = available.loc[available["inference_ms_per_sample"].idxmin()] if available["inference_ms_per_sample"].notna().any() else None

    lines = [
        "# Unified Model Benchmark",
        "",
        f"- Dataset: {dataset_name(args)}",
        f"- Train samples: {len(dataset.x_train)}",
        f"- Validation samples: {len(dataset.x_val)}",
        f"- Test samples: {len(dataset.x_test)}",
        f"- Timing samples: {min(args.timing_samples, len(dataset.x_test))}",
        f"- Timing repeats: {args.repeats}",
        f"- Normalization: {args.normalize}",
        "",
        "## Main comparison",
        "",
        "| model | family | method | epochs | accuracy | loss | params | sparsity | structured sparsity | size bytes | ms/sample | dataset |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['family']} | {row['method']} | {fmt(row['epochs'], 0)} | "
            f"{fmt(row['test_accuracy'])} | {fmt(row['test_loss'])} | {fmt(row['parameters'], 0)} | "
            f"{fmt(row['parameter_sparsity'])} | {fmt(row['structured_sparsity'])} | "
            f"{fmt(row['size_bytes'], 0)} | {fmt(row['inference_ms_per_sample'])} | {row['dataset']} |"
        )

    lines.extend(["", "## Summary interpretation", ""])
    if best_accuracy is not None:
        lines.append(f"- Best recorded test accuracy: `{best_accuracy['model']}` at {best_accuracy['test_accuracy']:.4f}.")
    if smallest is not None:
        lines.append(f"- Smallest recorded model artifact: `{smallest['model']}` at {int(smallest['size_bytes'])} bytes.")
    if fastest is not None:
        lines.append(f"- Fastest measured inference path in this run: `{fastest['model']}` at {fastest['inference_ms_per_sample']:.4f} ms/sample.")
    lines.extend(
        [
            "- Pruned model sizes are estimated from remaining nonzero float32 parameters unless a physically smaller architecture or sparse storage is used.",
            "- Missing cells mean the corresponding older artifact did not record that metric, not that the value is zero.",
            "- Timing is measured on this machine and should be reported with the hardware/software context.",
            "",
            "## Generated plots",
            "",
            "- `benchmark_accuracy.png`",
            "- `benchmark_size.png`",
            "- `benchmark_inference_time.png`",
            "- `benchmark_parameters.png`",
            "- `benchmark_sparsity.png`",
            "- `benchmark_accuracy_size_time.png`",
            "",
        ]
    )
    (args.output_dir / "model_benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def fmt(value, decimals: int = 4) -> str:
    if pd.isna(value):
        return ""
    if decimals == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def plot_benchmark_results(comparison: pd.DataFrame, output_dir: Path) -> None:
    available = comparison[comparison["result_available"] == True].copy()
    colors = ["#4c72b0", "#55a868", "#dd8452", "#8172b3", "#c44e52", "#64b5cd", "#937860"]
    plot_bar(available, "test_accuracy", "Test Accuracy", "Accuracy", output_dir / "benchmark_accuracy.png", colors, ylim=(0, 1))
    plot_bar(available, "size_bytes", "Model Size", "Bytes", output_dir / "benchmark_size.png", colors)
    plot_bar(available, "inference_ms_per_sample", "Inference Time", "Milliseconds per sample", output_dir / "benchmark_inference_time.png", colors)
    plot_bar(available, "parameters", "Parameter Count", "Parameters", output_dir / "benchmark_parameters.png", colors)

    sparsity = available[["model", "parameter_sparsity", "structured_sparsity"]].set_index("model")
    ax = sparsity.plot(kind="bar", figsize=(10, 5), color=["#4c72b0", "#dd8452"])
    ax.set_title("Parameter and Structured Sparsity")
    ax.set_ylabel("Sparsity")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_sparsity.png", dpi=180)
    plt.close()

    combo = available[["model", "test_accuracy", "size_bytes", "inference_ms_per_sample"]].copy()
    combo["size_kb"] = combo["size_bytes"] / 1024.0
    x = np.arange(len(combo))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    width = 0.35
    ax1.bar(x - width / 2, combo["test_accuracy"], width, color="#4c72b0", label="Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo["model"], rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, combo["size_kb"], width, color="#dd8452", label="Size KB")
    ax2.plot(x, combo["inference_ms_per_sample"], color="#55a868", marker="o", label="ms/sample")
    ax2.set_ylabel("Size KB / ms per sample")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Accuracy, Size, and Inference Time")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_accuracy_size_time.png", dpi=180)
    plt.close(fig)


def plot_bar(
    comparison: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    colors: list[str],
    ylim: tuple[float, float] | None = None,
) -> None:
    data = comparison[["model", column]].dropna()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(data["model"], data[column], color=colors[: len(data)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, data[column]):
        label = f"{value:.4f}" if abs(value) < 100 else f"{value:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
