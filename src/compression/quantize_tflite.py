from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a trained ECG model to int8 TensorFlow Lite.")
    parser.add_argument("--model", type=Path, default=Path("results/improved_cnn_scratch/tiny_ecg_cnn.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/improved_cnn_scratch/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--representative-samples", type=int, default=200)
    parser.add_argument("--demo-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = make_demo_dataset() if args.demo_data else load_mitbih_csv(args.data_dir, normalize=args.normalize)
    model = tf.keras.models.load_model(args.model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(dataset.x_train, args.representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    args.output.write_bytes(tflite_model)

    metrics = evaluate_tflite(args.output, dataset.x_test, dataset.y_test)
    metrics["tflite_size_bytes"] = args.output.stat().st_size
    metrics["tflite_path"] = str(args.output)
    metrics["representative_samples"] = min(args.representative_samples, len(dataset.x_train))
    baseline_metrics = _load_baseline_metrics(args.model.parent)
    if baseline_metrics is not None:
        metrics["baseline_test_accuracy"] = float(baseline_metrics["test_accuracy"])
        metrics["baseline_test_loss"] = float(baseline_metrics["test_loss"])
        metrics["baseline_size_bytes"] = int(
            baseline_metrics.get("keras_model_size_bytes", baseline_metrics.get("model_size_bytes", 0))
        )
        metrics["accuracy_delta"] = float(metrics["int8_accuracy"] - metrics["baseline_test_accuracy"])
        metrics["loss_delta"] = float(metrics["int8_loss"] - metrics["baseline_test_loss"])
        if metrics["baseline_size_bytes"]:
            metrics["size_reduction_bytes"] = int(metrics["baseline_size_bytes"] - metrics["tflite_size_bytes"])
            metrics["size_reduction_percent"] = float(
                100.0 * (1.0 - metrics["tflite_size_bytes"] / metrics["baseline_size_bytes"])
            )

    (args.output.parent / "int8_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_quantization_report(metrics, args.output.parent)
    print(json.dumps(metrics, indent=2))


def representative_dataset(x_train: np.ndarray, max_samples: int):
    def generator():
        for sample in x_train[:max_samples]:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    return generator


def evaluate_tflite(model_path: Path, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    correct = 0
    probabilities = []
    predictions = []
    for x, y in zip(x_test, y_test):
        quantized_x = x / input_scale + input_zero_point
        quantized_x = np.clip(quantized_x, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized_x[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        if output_scale:
            output_float = (output.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output.astype(np.float32)
        prediction = int(output_float.argmax(axis=1)[0])
        correct += int(prediction == y)
        predictions.append(prediction)
        probabilities.append(np.clip(output_float[0], 1e-12, 1.0))

    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    loss = sparse_cross_entropy_from_probabilities(probabilities_array, y_test)
    report = classification_report(y_test, predictions_array, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions_array)
    model_dir = model_path.parent
    (model_dir / "int8_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(model_dir / "int8_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    return {"int8_accuracy": correct / len(y_test), "int8_loss": loss}


def sparse_cross_entropy_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    correct_log_probs = -np.log(probabilities[np.arange(len(labels)), labels] + 1e-12)
    return float(np.mean(correct_log_probs))


def _load_baseline_metrics(model_dir: Path) -> dict | None:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def write_quantization_report(metrics: dict, output_dir: Path) -> None:
    if "baseline_test_accuracy" in metrics:
        plot_accuracy_comparison(metrics, output_dir / "cnn_quantization_accuracy_comparison.png")
        plot_size_comparison(metrics, output_dir / "cnn_quantization_size_comparison.png")

    lines = [
        "# CNN Post-Training Quantization Summary",
        "",
        "This report compares the improved CNN Keras model against a post-training int8 TensorFlow Lite model.",
        "",
        "## Quantized model",
        "",
        f"- Int8 accuracy: {metrics['int8_accuracy']:.4f}",
        f"- Int8 loss: {metrics['int8_loss']:.4f}",
        f"- TFLite size: {metrics['tflite_size_bytes']} bytes",
        f"- TFLite path: `{metrics['tflite_path']}`",
        f"- Representative samples: {metrics['representative_samples']}",
    ]

    if "baseline_test_accuracy" in metrics:
        lines.extend([
            "",
            "## Baseline vs int8",
            "",
            "| metric | baseline CNN | int8 TFLite CNN | change |",
            "|---|---:|---:|---:|",
            f"| accuracy | {metrics['baseline_test_accuracy']:.4f} | {metrics['int8_accuracy']:.4f} | {metrics['accuracy_delta']:+.4f} |",
            f"| loss | {metrics['baseline_test_loss']:.4f} | {metrics['int8_loss']:.4f} | {metrics['loss_delta']:+.4f} |",
            f"| size bytes | {metrics['baseline_size_bytes']} | {metrics['tflite_size_bytes']} | -{metrics.get('size_reduction_bytes', 0)} |",
            "",
            f"- Size reduction: {metrics.get('size_reduction_percent', 0.0):.2f}%",
            "",
            "Generated plots:",
            "",
            "- `cnn_quantization_accuracy_comparison.png`",
            "- `cnn_quantization_size_comparison.png`",
        ])

    lines.extend([
        "",
        "Generated evaluation artifacts:",
        "",
        "- `int8_metrics.json`",
        "- `int8_classification_report.json`",
        "- `int8_confusion_matrix.csv`",
        "",
    ])
    (output_dir / "cnn_quantization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_accuracy_comparison(metrics: dict, output_path: Path) -> None:
    labels = ["Baseline", "PTQ int8"]
    values = [metrics["baseline_test_accuracy"], metrics["int8_accuracy"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452"])
    ax.set_title("CNN Baseline vs PTQ Int8 Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_size_comparison(metrics: dict, output_path: Path) -> None:
    labels = ["Baseline", "PTQ int8"]
    values = [metrics["baseline_size_bytes"] / 1024.0, metrics["tflite_size_bytes"] / 1024.0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452"])
    ax.set_title("CNN Baseline vs PTQ Int8 Size")
    ax.set_ylabel("Size (KB)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f} KB", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
