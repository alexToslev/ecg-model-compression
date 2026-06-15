from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.mitbih_csv import load_mitbih_csv, make_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one ECG test sample as a C header for ESP32 inference.")
    parser.add_argument("--model", type=Path, default=Path("results/quantized/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("esp32/ecg_tflite_micro/test_sample.h"))
    parser.add_argument("--sample-index", type=int, default=0)
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
    if args.sample_index < 0 or args.sample_index >= len(dataset.x_test):
        raise IndexError(f"--sample-index must be between 0 and {len(dataset.x_test) - 1}")

    sample = dataset.x_test[args.sample_index]
    expected_label = int(dataset.y_test[args.sample_index])
    input_scale, input_zero_point = read_input_quantization(args.model)
    quantized_sample = quantize_sample(sample, input_scale, input_zero_point)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        make_header(
            quantized_sample=quantized_sample,
            expected_label=expected_label,
            sample_index=args.sample_index,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
        ),
        encoding="utf-8",
    )

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "sample_index": args.sample_index,
        "expected_label": expected_label,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "length": int(quantized_sample.size),
        "source": "demo-data" if args.demo_data else str(args.data_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote sample header to {args.output}")
    print(f"Wrote sample metadata to {metadata_path}")
    print(json.dumps(metadata, indent=2))


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
    quantized_sample: np.ndarray,
    expected_label: int,
    sample_index: int,
    input_scale: float,
    input_zero_point: int,
) -> str:
    values = [str(int(value)) for value in quantized_sample]
    rows = []
    for start in range(0, len(values), 16):
        rows.append("  " + ", ".join(values[start : start + 16]) + ",")

    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            f"const int g_ecg_sample_index = {sample_index};",
            f"const int g_ecg_expected_label = {expected_label};",
            f"const float g_ecg_input_scale = {input_scale:.10g}f;",
            f"const int g_ecg_input_zero_point = {input_zero_point};",
            f"const int g_ecg_sample_len = {quantized_sample.size};",
            "const int8_t g_ecg_sample[] = {",
            *rows,
            "};",
            "",
        ]
    )


if __name__ == "__main__":
    main()
