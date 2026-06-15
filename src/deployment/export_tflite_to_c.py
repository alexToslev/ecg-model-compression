from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a TFLite file as a C header for Arduino/ESP32.")
    parser.add_argument("--model", type=Path, default=Path("results/quantized/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--output", type=Path, default=Path("esp32/ecg_tflite_micro/model_data.h"))
    parser.add_argument("--array-name", default="g_ecg_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"TFLite model not found: {args.model}")

    model_bytes = args.model.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        make_header(
            data=model_bytes,
            array_name=args.array_name,
            source_path=args.model,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(model_bytes)} bytes to {args.output}")


def make_header(data: bytes, array_name: str, source_path: Path) -> str:
    hex_values = [f"0x{byte:02x}" for byte in data]
    rows = []
    for start in range(0, len(hex_values), 12):
        rows.append("  " + ", ".join(hex_values[start : start + 12]) + ",")

    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            f"// Generated from: {source_path.as_posix()}",
            f"alignas(16) const unsigned char {array_name}[] = {{",
            *rows,
            "};",
            f"const int {array_name}_len = {len(data)};",
            "",
        ]
    )


if __name__ == "__main__":
    main()
