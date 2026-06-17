from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ECG test samples as a C header for ESP32 inference.")
    parser.add_argument("--model", type=Path, default=Path("results/quantized/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("esp32/ecg_tflite_micro/test_sample.h"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--normalize", choices=["none", "standard", "per_sample"], default="none")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"TFLite model not found: {args.model}")

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
    sample_indices = choose_sample_indices(
        labels=dataset.y_test,
        sample_index=args.sample_index,
        sample_count=args.sample_count,
        samples_per_class=args.samples_per_class,
    )
    input_scale, input_zero_point = read_input_quantization(args.model)
    quantized_samples = np.stack(
        [quantize_sample(dataset.x_test[index], input_scale, input_zero_point) for index in sample_indices]
    )
    expected_labels = [int(dataset.y_test[index]) for index in sample_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        make_header(
            quantized_samples=quantized_samples,
            expected_labels=expected_labels,
            sample_indices=sample_indices,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
        ),
        encoding="utf-8",
    )

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "sample_indices": sample_indices,
        "expected_labels": expected_labels,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "sample_count": len(sample_indices),
        "sample_length": int(quantized_samples.shape[1]),
        "source": "demo-data" if args.demo_data else str(args.data_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote sample header to {args.output}")
    print(f"Wrote sample metadata to {metadata_path}")
    print(json.dumps(metadata, indent=2))


def choose_sample_indices(
    labels: np.ndarray,
    sample_index: int,
    sample_count: int,
    samples_per_class: int,
) -> list[int]:
    if samples_per_class > 0:
        chosen = []
        for label in sorted(np.unique(labels).astype(int)):
            class_indices = np.flatnonzero(labels == label)
            chosen.extend(int(index) for index in class_indices[:samples_per_class])
        return chosen

    if sample_count < 1:
        raise ValueError("--sample-count must be at least 1")
    if sample_index < 0 or sample_index >= len(labels):
        raise IndexError(f"--sample-index must be between 0 and {len(labels) - 1}")

    last_index = sample_index + sample_count
    if last_index > len(labels):
        raise IndexError(f"Requested samples end at {last_index - 1}, but the last index is {len(labels) - 1}")
    return list(range(sample_index, last_index))


def read_input_quantization(model_path: Path) -> tuple[float, int]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    return float(scale), int(zero_point)


def quantize_sample(sample: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    values = sample.reshape(-1).astype(np.float32)
    quantized = np.rint(values / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def make_header(
    quantized_samples: np.ndarray,
    expected_labels: list[int],
    sample_indices: list[int],
    input_scale: float,
    input_zero_point: int,
) -> str:
    sample_rows = []
    for sample in quantized_samples:
        values = [str(int(value)) for value in sample]
        sample_rows.append("  {")
        for start in range(0, len(values), 16):
            sample_rows.append("    " + ", ".join(values[start : start + 16]) + ",")
        sample_rows.append("  },")

    label_values = ", ".join(str(label) for label in expected_labels)
    index_values = ", ".join(str(index) for index in sample_indices)

    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            f"const int g_ecg_num_samples = {len(sample_indices)};",
            f"const int g_ecg_sample_len = {quantized_samples.shape[1]};",
            f"const float g_ecg_input_scale = {input_scale:.10g}f;",
            f"const int g_ecg_input_zero_point = {input_zero_point};",
            f"const int g_ecg_sample_indices[] = {{{index_values}}};",
            f"const int g_ecg_expected_labels[] = {{{label_values}}};",
            "const int8_t g_ecg_samples[][g_ecg_sample_len] = {",
            *sample_rows,
            "};",
            "",
        ]
    )


if __name__ == "__main__":
    main()
