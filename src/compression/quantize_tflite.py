from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a trained ECG model to int8 TensorFlow Lite.")
    parser.add_argument("--model", type=Path, default=Path("results/baseline_cnn/tiny_ecg_cnn.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/quantized/tiny_ecg_cnn_int8.tflite"))
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
    (args.output.parent / "int8_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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

    correct = 0
    for x, y in zip(x_test, y_test):
        quantized_x = x / input_scale + input_zero_point
        quantized_x = np.clip(quantized_x, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized_x[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        prediction = int(output.argmax(axis=1)[0])
        correct += int(prediction == y)

    return {"int8_accuracy": correct / len(y_test)}


if __name__ == "__main__":
    main()
