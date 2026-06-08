from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset
from src.models.cnn1d import build_tiny_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert and evaluate an ECG model as full-int8 TensorFlow Lite.")
    parser.add_argument("--model", type=Path, default=Path("results/baseline_cnn/tiny_ecg_cnn.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/quantized"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--representative-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = (
        make_demo_dataset(validation_fraction=args.validation_fraction, seed=args.seed)
        if args.demo_data
        else load_mitbih_csv(
            data_dir=args.data_dir,
            validation_fraction=args.validation_fraction,
            normalize=args.normalize,
            seed=args.seed,
        )
    )
    model = load_or_train_model(args, dataset)

    float_metrics, float_predictions = evaluate_keras_model(model, dataset, args.batch_size)
    tflite_path = args.output or args.output_dir / "tiny_ecg_cnn_int8.tflite"
    convert_to_int8_tflite(model, dataset.x_train, tflite_path, args.representative_samples)
    int8_metrics, int8_predictions, quantization_details = evaluate_tflite_model(
        tflite_path,
        dataset.x_test,
        dataset.y_test,
    )

    summary = save_results(
        output_dir=args.output_dir,
        model_path=args.model,
        tflite_path=tflite_path,
        model=model,
        dataset=dataset,
        float_metrics=float_metrics,
        int8_metrics=int8_metrics,
        float_predictions=float_predictions,
        int8_predictions=int8_predictions,
        quantization_details=quantization_details,
        representative_samples=args.representative_samples,
    )
    print(json.dumps(summary, indent=2))


def load_or_train_model(args: argparse.Namespace, dataset) -> tf.keras.Model:
    if args.model.exists():
        return tf.keras.models.load_model(args.model)

    if not args.train_if_missing:
        raise FileNotFoundError(
            f"Model not found at {args.model}. Train it first with src.train_cnn, "
            "or pass --train-if-missing for a quick experiment."
        )

    model = build_tiny_cnn(
        input_length=dataset.input_length,
        num_classes=dataset.num_classes,
        learning_rate=args.learning_rate,
    )
    model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    return model


def convert_to_int8_tflite(
    model: tf.keras.Model,
    x_train: np.ndarray,
    output_path: Path,
    representative_samples: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(x_train, representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    output_path.write_bytes(converter.convert())


def representative_dataset(x_train: np.ndarray, max_samples: int):
    def generator():
        for sample in x_train[:max_samples]:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    return generator


def evaluate_keras_model(model: tf.keras.Model, dataset, batch_size: int) -> tuple[dict[str, float], np.ndarray]:
    loss, accuracy = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    probabilities = model.predict(dataset.x_test, batch_size=batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return {"loss": float(loss), "accuracy": float(accuracy)}, predictions


def evaluate_tflite_model(
    model_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, dict[str, object]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    predictions: list[int] = []
    for x in x_test:
        quantized_x = x / input_scale + input_zero_point
        quantized_x = np.clip(np.rint(quantized_x), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized_x[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        predictions.append(int(output.argmax(axis=1)[0]))

    predictions_array = np.asarray(predictions, dtype=np.int64)
    accuracy = float((predictions_array == y_test).mean())
    quantization_details = {
        "input_dtype": str(input_details["dtype"]),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_dtype": str(output_details["dtype"]),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }
    return {"accuracy": accuracy}, predictions_array, quantization_details


def save_results(
    output_dir: Path,
    model_path: Path,
    tflite_path: Path,
    model: tf.keras.Model,
    dataset,
    float_metrics: dict[str, float],
    int8_metrics: dict[str, float],
    float_predictions: np.ndarray,
    int8_predictions: np.ndarray,
    quantization_details: dict[str, object],
    representative_samples: int,
) -> dict[str, object]:
    keras_size = model_path.stat().st_size if model_path.exists() else None
    tflite_size = tflite_path.stat().st_size
    accuracy_drop = float_metrics["accuracy"] - int8_metrics["accuracy"]
    size_reduction_factor = (keras_size / tflite_size) if keras_size else None

    summary = {
        "float32_model_path": str(model_path),
        "int8_tflite_path": str(tflite_path),
        "input_length": dataset.input_length,
        "num_classes": dataset.num_classes,
        "parameters": int(model.count_params()),
        "float32_accuracy": float_metrics["accuracy"],
        "float32_loss": float_metrics["loss"],
        "int8_accuracy": int8_metrics["accuracy"],
        "accuracy_drop": accuracy_drop,
        "keras_size_bytes": keras_size,
        "tflite_size_bytes": tflite_size,
        "size_reduction_factor": size_reduction_factor,
        "representative_samples": representative_samples,
        "quantization": quantization_details,
    }
    (output_dir / "quantization_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {"model": "float32_keras", "accuracy": float_metrics["accuracy"], "size_bytes": keras_size},
            {"model": "int8_tflite", "accuracy": int8_metrics["accuracy"], "size_bytes": tflite_size},
        ]
    ).to_csv(output_dir / "float32_vs_int8.csv", index=False)

    float_report = classification_report(dataset.y_test, float_predictions, output_dict=True, zero_division=0)
    int8_report = classification_report(dataset.y_test, int8_predictions, output_dict=True, zero_division=0)
    (output_dir / "float32_classification_report.json").write_text(
        json.dumps(float_report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "int8_classification_report.json").write_text(
        json.dumps(int8_report, indent=2),
        encoding="utf-8",
    )
    np.savetxt(output_dir / "int8_confusion_matrix.csv", confusion_matrix(dataset.y_test, int8_predictions), delimiter=",", fmt="%d")
    write_markdown_summary(output_dir / "quantization_summary.md", summary)
    return summary


def write_markdown_summary(output_path: Path, summary: dict[str, object]) -> None:
    keras_size = summary["keras_size_bytes"]
    tflite_size = summary["tflite_size_bytes"]
    size_text = "not available"
    if keras_size:
        size_text = f"{keras_size:,} bytes -> {tflite_size:,} bytes ({summary['size_reduction_factor']:.2f}x smaller)"

    lines = [
        "# Quantization Summary",
        "",
        "## Intuition",
        "",
        "Post-training int8 quantization stores model weights and activations as 8-bit integers instead of 32-bit floating point numbers. "
        "This usually makes the model smaller and more microcontroller-friendly, but rounding can slightly change the predictions.",
        "",
        "## Formal View",
        "",
        "TensorFlow Lite represents a real tensor value using an integer value, a scale, and a zero-point:",
        "",
        "`real_value = scale * (integer_value - zero_point)`",
        "",
        "The representative dataset is used during conversion to estimate useful scales for ECG inputs and intermediate activations.",
        "",
        "## Results",
        "",
        f"- Parameters: {summary['parameters']}",
        f"- Input length/classes: {summary['input_length']} / {summary['num_classes']}",
        f"- Float32 accuracy: {summary['float32_accuracy']:.4f}",
        f"- Int8 accuracy: {summary['int8_accuracy']:.4f}",
        f"- Accuracy drop: {summary['accuracy_drop']:.4f}",
        f"- Size: {size_text}",
        f"- Representative samples: {summary['representative_samples']}",
        "",
        "## ESP32 Feasibility",
        "",
        "The `.tflite` file size is only the first check. Before deployment, also check TensorFlow Lite Micro operator support, "
        "RAM for intermediate tensors, C-array export, and real on-device inference time.",
        "",
        "## Class Imbalance Note",
        "",
        "For MIT-BIH, total accuracy is not enough. Always inspect the per-class reports and confusion matrix, because minority-class recall can be weak even when overall accuracy is high.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    main()
