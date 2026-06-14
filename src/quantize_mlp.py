from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.mlp import Dense, ManualMLP, ReLU, build_manual_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a manual MLP to 8-bit fixed-point and compare against the original model."
    )
    parser.add_argument("--weights", type=Path, default=Path("results/baseline_mlp/baseline_mlp_weights.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_mlp/quantization"))
    parser.add_argument("--demo-data", action="store_true")
    parser.add_argument(
        "--normalize",
        choices=["none", "standard", "per_sample"],
        default="none",
    )
    parser.add_argument("--calibration-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = (
        make_demo_dataset()
        if args.demo_data
        else load_mitbih_csv(
            data_dir=args.data_dir,
            normalize=args.normalize,
        )
    )

    float_model = build_manual_mlp(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
    )
    float_model.set_parameters(_load_weights(args.weights))

    original_metrics = _evaluate_model(float_model, dataset.x_test, dataset.y_test)

    calibration_inputs = dataset.x_train[: args.calibration_samples]
    q_model = QuantizedManualMLP.from_float_model(float_model, calibration_inputs)
    quantized_metrics = _evaluate_quantized_model(q_model, dataset.x_test, dataset.y_test)

    metrics = {
        "original_test_loss": original_metrics[0],
        "original_test_accuracy": original_metrics[1],
        "quantized_test_loss": quantized_metrics[0],
        "quantized_test_accuracy": quantized_metrics[1],
        "original_model_size_bytes": q_model.original_size_bytes,
        "quantized_model_size_bytes": q_model.quantized_size_bytes,
        "compression_ratio": float(q_model.original_size_bytes / q_model.quantized_size_bytes),
        "input_scale": float(q_model.input_scale),
        "activation_scales": [float(scale) for scale in q_model.activation_scales],
        "weight_scales": [float(layer.weight_scale) for layer in q_model.dense_layers],
        "bias_scales": [float(layer.bias_scale) for layer in q_model.dense_layers],
    }

    summary_path = args.output_dir / "quantization_metrics.json"
    summary_csv_path = args.output_dir / "quantization_metrics.csv"
    pd.DataFrame([metrics]).to_csv(summary_csv_path, index=False)
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_quantized_model(q_model, args.output_dir / "baseline_mlp_quantized.npz")
    plot_quantization_results(metrics, args.output_dir)
    write_quantization_summary(metrics, args.output_dir)

    print(json.dumps(metrics, indent=2))
    print(f"[quantize_mlp] saved quantization metrics to {summary_path}")


def _load_weights(weights_path: Path) -> dict[str, np.ndarray]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    with np.load(weights_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _evaluate_model(model: ManualMLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(x)
    loss = model.loss.forward(logits, y)
    predictions = np.argmax(logits, axis=1)
    return float(loss), float(np.mean(predictions == y))


def _evaluate_quantized_model(model: "QuantizedManualMLP", x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(x)
    loss = model.loss.forward(logits, y)
    predictions = np.argmax(logits, axis=1)
    return float(loss), float(np.mean(predictions == y))


def quantize_tensor(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    max_val = float(np.max(np.abs(tensor)))
    scale = max(max_val / 127.0, 1e-8)
    quantized = np.round(tensor / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def quantize_activation(x_float: np.ndarray, scale: float) -> np.ndarray:
    return np.round(x_float / scale).clip(-127, 127).astype(np.int8)


class QuantizedDense:
    def __init__(self, weights_int8: np.ndarray, weight_scale: float, bias_int32: np.ndarray, output_scale: float, bias_scale: float):
        self.weights_int8 = weights_int8
        self.weight_scale = weight_scale
        self.bias_int32 = bias_int32
        self.output_scale = output_scale
        self.bias_scale = bias_scale

    def forward(self, x_int8: np.ndarray) -> np.ndarray:
        return x_int8.astype(np.int32) @ self.weights_int8.astype(np.int32) + self.bias_int32


class QuantizedManualMLP:
    def __init__(
        self,
        dense_layers: list[QuantizedDense],
        input_scale: float,
        activation_scales: list[float],
        original_size_bytes: int,
        quantized_size_bytes: int,
    ):
        self.dense_layers = dense_layers
        self.input_scale = input_scale
        self.activation_scales = activation_scales
        self.original_size_bytes = original_size_bytes
        self.quantized_size_bytes = quantized_size_bytes
        self.loss = SoftmaxCrossEntropy()

    @classmethod
    def from_float_model(cls, model: ManualMLP, x_calibration: np.ndarray) -> "QuantizedManualMLP":
        input_scale = _calibrate_input_scale(x_calibration)
        activation_scales = _calibrate_activation_scales(model, x_calibration)

        quantized_layers: list[QuantizedDense] = []
        input_scale_for_layer = input_scale
        original_bytes = 0
        quantized_bytes = 0

        hidden_dense_count = sum(1 for layer in model.layers if isinstance(layer, Dense))
        dense_index = 0

        for layer in model.layers + [model.output_layer]:
            if not isinstance(layer, Dense):
                continue

            weight_int8, weight_scale = quantize_tensor(layer.weights)
            output_scale = input_scale_for_layer * weight_scale
            bias_scale = input_scale_for_layer * weight_scale
            bias_int32 = np.round(layer.bias / bias_scale).astype(np.int32)
            quantized_layers.append(QuantizedDense(weight_int8, weight_scale, bias_int32, output_scale, bias_scale))

            original_bytes += int(layer.weights.size * 4 + layer.bias.size * 4)
            quantized_bytes += int(layer.weights.size * 1 + layer.bias.size * 4 + 2 * 4)
            if dense_index < len(activation_scales):
                input_scale_for_layer = activation_scales[dense_index]
            dense_index += 1

        return cls(
            dense_layers=quantized_layers,
            input_scale=input_scale,
            activation_scales=activation_scales,
            original_size_bytes=original_bytes,
            quantized_size_bytes=quantized_bytes,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], -1)
        x_int8 = quantize_activation(x, self.input_scale)
        current_scale = self.input_scale

        dense_idx = 0
        for activation_index, dense_layer in enumerate(self.dense_layers):
            x_int32 = dense_layer.forward(x_int8)
            x_float = x_int32.astype(np.float32) * dense_layer.output_scale

            if dense_idx < len(self.activation_scales):
                x_float = np.maximum(x_float, 0.0)
                activation_scale = self.activation_scales[dense_idx]
                x_int8 = quantize_activation(x_float, activation_scale)
                current_scale = activation_scale
                dense_idx += 1
            else:
                return x_float

        return x_float

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probabilities: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.labels = labels
        correct_log_probs = -np.log(self.probabilities[np.arange(len(labels)), labels] + 1e-12)
        return float(np.mean(correct_log_probs))


def _calibrate_input_scale(x_calibration: np.ndarray) -> float:
    max_val = float(np.max(np.abs(x_calibration)))
    return max(max_val / 127.0, 1e-8)


def _calibrate_activation_scales(model: ManualMLP, x_calibration: np.ndarray) -> list[float]:
    x = model.flatten.forward(x_calibration)
    scales: list[float] = []
    for layer in model.layers:
        x = layer.forward(x)
        if isinstance(layer, ReLU):
            max_val = float(np.max(np.abs(x)))
            scales.append(max(max_val / 127.0, 1e-8))
    return scales


def save_quantized_model(model: QuantizedManualMLP, path: Path) -> None:
    arrays = {
        f"weights_{i}": layer.weights_int8 for i, layer in enumerate(model.dense_layers)
    }
    arrays.update({f"bias_{i}": layer.bias_int32 for i, layer in enumerate(model.dense_layers)})
    arrays["input_scale"] = np.array([model.input_scale], dtype=np.float32)
    arrays["activation_scales"] = np.array(model.activation_scales, dtype=np.float32)
    arrays["weight_scales"] = np.array([layer.weight_scale for layer in model.dense_layers], dtype=np.float32)
    arrays["bias_scales"] = np.array([layer.bias_scale for layer in model.dense_layers], dtype=np.float32)
    np.savez(path, **arrays)


def plot_quantization_results(metrics: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["original", "quantized"], [metrics["original_test_accuracy"], metrics["quantized_test_accuracy"]], color=["#4c72b0", "#dd8452"])
    ax.set_title("Original vs Quantized Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "quantization_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["original", "quantized"], [metrics["original_model_size_bytes"], metrics["quantized_model_size_bytes"]], color=["#4c72b0", "#dd8452"])
    ax.set_title("Original vs Quantized Model Size")
    ax.set_ylabel("Bytes")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "quantization_size_comparison.png", dpi=180)
    plt.close(fig)


def write_quantization_summary(metrics: dict, output_dir: Path) -> None:
    lines = [
        "# Manual MLP Quantization Summary",
        "",
        "This report compares the original manual MLP against an 8-bit fixed-point quantized version.",
        "",
        f"- Original test accuracy: {metrics['original_test_accuracy']:.4f}",
        f"- Quantized test accuracy: {metrics['quantized_test_accuracy']:.4f}",
        f"- Original test loss: {metrics['original_test_loss']:.4f}",
        f"- Quantized test loss: {metrics['quantized_test_loss']:.4f}",
        f"- Original model size (bytes): {metrics['original_model_size_bytes']}",
        f"- Quantized model size (bytes): {metrics['quantized_model_size_bytes']}",
        f"- Compression ratio: {metrics['compression_ratio']:.4f}",
        "",
        "## Generated plots",
        "",
        "- `quantization_accuracy_comparison.png`",
        "- `quantization_size_comparison.png`",
        "",
        "## Notes",
        "",
        "- The quantized model uses symmetric int8 weights and int32 bias accumulation.",
        "- Fixed-point inference uses calibrated input and hidden activation scales.",
    ]
    (output_dir / "quantization_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
