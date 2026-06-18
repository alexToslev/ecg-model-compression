from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an int8 ECG CNN TFLite model for TinyML/ESP32 simulation and deployment planning."
    )
    parser.add_argument("--model", type=Path, default=Path("results/improved_cnn/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--metrics", type=Path, default=Path("results/improved_cnn/int8_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/esp32"))
    parser.add_argument("--model-name", type=str, default="improved_ecg_cnn_int8")
    parser.add_argument("--arena-bytes", type=int, default=None)
    parser.add_argument("--flash-budget-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--sram-budget-bytes", type=int, default=320 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"TFLite model not found: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = args.model.read_bytes()
    model_info = inspect_tflite_model(args.model)
    metrics = load_json(args.metrics)

    arena_bytes = args.arena_bytes or estimate_tensor_arena_bytes(model_info)
    decision = deployment_decision(
        model_size=len(model_bytes),
        arena_bytes=arena_bytes,
        flash_budget=args.flash_budget_bytes,
        sram_budget=args.sram_budget_bytes,
    )

    header_path = args.output_dir / f"{args.model_name}_model.h"
    sketch_path = args.output_dir / "esp32_tflite_micro_example.ino"
    report_json_path = args.output_dir / "esp32_deployment_report.json"
    report_md_path = args.output_dir / "esp32_deployment_report.md"

    write_model_header(model_bytes, args.model_name, header_path)
    write_arduino_sketch(args.model_name, arena_bytes, sketch_path)

    report = {
        "decision": decision,
        "model_path": str(args.model),
        "model_size_bytes": len(model_bytes),
        "model_array_header": str(header_path),
        "example_sketch": str(sketch_path),
        "estimated_tensor_arena_bytes": arena_bytes,
        "estimated_total_ram_bytes": arena_bytes + model_info.get("io_tensor_bytes", 0),
        "flash_budget_bytes": args.flash_budget_bytes,
        "sram_budget_bytes": args.sram_budget_bytes,
        "input": model_info.get("input"),
        "output": model_info.get("output"),
        "operators": model_info.get("operators", []),
        "tensor_bytes_observed_by_interpreter": model_info.get("tensor_bytes"),
        "io_tensor_bytes": model_info.get("io_tensor_bytes"),
        "int8_metrics": metrics,
        "notes": [
            "This is an export and simulation report, not proof of a flashed ESP32 run.",
            "Tensor arena is estimated from the local TFLite interpreter tensor shapes; final firmware may need tuning.",
            "Use the generated header and sketch as a starting point for TensorFlow Lite Micro on ESP32.",
        ],
    }

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, report_md_path)
    print(json.dumps(report, indent=2))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def inspect_tflite_model(model_path: Path) -> dict:
    try:
        import tensorflow as tf
    except ImportError:
        return {
            "input": None,
            "output": None,
            "operators": [],
            "tensor_bytes": None,
            "io_tensor_bytes": 0,
            "inspection_note": "TensorFlow is not installed, so interpreter-level memory inspection was skipped.",
        }

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    tensor_details = interpreter.get_tensor_details()

    tensor_bytes = 0
    for tensor in tensor_details:
        shape = tensor.get("shape", [])
        dtype = np.dtype(tensor["dtype"])
        element_count = int(np.prod(shape)) if len(shape) else 1
        tensor_bytes += element_count * dtype.itemsize

    io_tensor_bytes = tensor_nbytes(input_details) + tensor_nbytes(output_details)
    operators = []
    if hasattr(interpreter, "_get_ops_details"):
        operators = sorted(
            {
                op.get("op_name", "UNKNOWN")
                for op in interpreter._get_ops_details()
                if op.get("op_name") != "DELEGATE"
            }
        )

    return {
        "input": tensor_summary(input_details),
        "output": tensor_summary(output_details),
        "operators": operators,
        "tensor_bytes": int(tensor_bytes),
        "io_tensor_bytes": int(io_tensor_bytes),
    }


def tensor_summary(tensor: dict) -> dict:
    scale, zero_point = tensor.get("quantization", (0.0, 0))
    return {
        "name": tensor.get("name", ""),
        "shape": [int(value) for value in tensor.get("shape", [])],
        "dtype": str(np.dtype(tensor["dtype"])),
        "quantization_scale": float(scale),
        "quantization_zero_point": int(zero_point),
        "bytes": tensor_nbytes(tensor),
    }


def tensor_nbytes(tensor: dict) -> int:
    shape = tensor.get("shape", [])
    dtype = np.dtype(tensor["dtype"])
    element_count = int(np.prod(shape)) if len(shape) else 1
    return int(element_count * dtype.itemsize)


def estimate_tensor_arena_bytes(model_info: dict) -> int:
    tensor_bytes = model_info.get("tensor_bytes")
    if tensor_bytes is None:
        return 64 * 1024
    # TFLite Micro needs planner overhead beyond raw tensors. Keep the estimate simple and conservative.
    return int(max(64 * 1024, round(tensor_bytes * 2.5 / 1024) * 1024))


def deployment_decision(model_size: int, arena_bytes: int, flash_budget: int, sram_budget: int) -> str:
    flash_ok = model_size < flash_budget * 0.75
    ram_ok = arena_bytes < sram_budget * 0.75
    if flash_ok and ram_ok:
        return "export_ready_for_esp32_simulation"
    if flash_ok:
        return "flash_ok_but_ram_needs_hardware_tuning"
    return "model_too_large_for_default_esp32_budget"


def write_model_header(model_bytes: bytes, model_name: str, output_path: Path) -> None:
    array_name = f"g_{model_name}_model"
    length_name = f"g_{model_name}_model_len"
    hex_values = [f"0x{value:02x}" for value in model_bytes]
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#if defined(ARDUINO)",
        "#include <Arduino.h>",
        "#else",
        "#define PROGMEM",
        "#endif",
        "",
        f"alignas(16) const unsigned char {array_name}[] PROGMEM = {{",
    ]
    for start in range(0, len(hex_values), 12):
        lines.append("  " + ", ".join(hex_values[start : start + 12]) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int {length_name} = {len(model_bytes)};",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_arduino_sketch(model_name: str, arena_bytes: int, output_path: Path) -> None:
    array_name = f"g_{model_name}_model"
    header_name = f"{model_name}_model.h"
    sketch = f"""
    #include <TensorFlowLite_ESP32.h>
    #include "tensorflow/lite/micro/all_ops_resolver.h"
    #include "tensorflow/lite/micro/micro_interpreter.h"
    #include "tensorflow/lite/schema/schema_generated.h"
    #include "tensorflow/lite/version.h"
    #include "{header_name}"

    namespace {{
    constexpr int kTensorArenaSize = {arena_bytes};
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    tflite::AllOpsResolver resolver;
    }}

    void setup() {{
      Serial.begin(115200);
      model = tflite::GetModel({array_name});
      if (model->version() != TFLITE_SCHEMA_VERSION) {{
        Serial.println("TFLite schema mismatch");
        return;
      }}

      static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
      );
      interpreter = &static_interpreter;

      if (interpreter->AllocateTensors() != kTfLiteOk) {{
        Serial.println("AllocateTensors failed; increase kTensorArenaSize");
        return;
      }}

      input = interpreter->input(0);
      output = interpreter->output(0);
      Serial.println("ECG CNN model loaded on ESP32");
    }}

    void loop() {{
      if (input == nullptr || output == nullptr) {{
        delay(1000);
        return;
      }}

      // Replace this zero-filled sample with one normalized MIT-BIH heartbeat of length 187.
      for (int i = 0; i < input->bytes; ++i) {{
        input->data.int8[i] = input->params.zero_point;
      }}

      unsigned long start_us = micros();
      TfLiteStatus status = interpreter->Invoke();
      unsigned long elapsed_us = micros() - start_us;

      if (status != kTfLiteOk) {{
        Serial.println("Inference failed");
        delay(1000);
        return;
      }}

      int best_class = 0;
      int8_t best_score = output->data.int8[0];
      for (int i = 1; i < output->bytes; ++i) {{
        if (output->data.int8[i] > best_score) {{
          best_score = output->data.int8[i];
          best_class = i;
        }}
      }}

      Serial.print("Predicted ECG class: ");
      Serial.print(best_class);
      Serial.print(" inference_us=");
      Serial.println(elapsed_us);
      delay(1000);
    }}
    """
    output_path.write_text(textwrap.dedent(sketch).strip() + "\n", encoding="utf-8")


def write_markdown_report(report: dict, output_path: Path) -> None:
    metrics = report.get("int8_metrics", {})
    input_info = report.get("input") or {}
    output_info = report.get("output") or {}
    operators = report.get("operators") or []

    lines = [
        "# ESP32 / TinyML Deployment Simulation Report",
        "",
        "## Decision",
        "",
        f"- Deployment status: `{report['decision']}`",
        "- Hardware execution was not performed in this WP; this report prepares the model for ESP32/TinyML testing.",
        "",
        "## Exported files",
        "",
        f"- C model header: `{report['model_array_header']}`",
        f"- Example ESP32 sketch: `{report['example_sketch']}`",
        f"- Source TFLite model: `{report['model_path']}`",
        "",
        "## Model footprint",
        "",
        f"- TFLite model size: {report['model_size_bytes']} bytes",
        f"- Estimated tensor arena: {report['estimated_tensor_arena_bytes']} bytes",
        f"- Estimated total RAM for arena plus I/O tensors: {report['estimated_total_ram_bytes']} bytes",
        f"- Assumed flash budget: {report['flash_budget_bytes']} bytes",
        f"- Assumed SRAM budget: {report['sram_budget_bytes']} bytes",
        "",
        "## Model interface",
        "",
        f"- Input shape: {input_info.get('shape', '')}",
        f"- Input dtype: {input_info.get('dtype', '')}",
        f"- Input quantization: scale={input_info.get('quantization_scale', '')}, zero_point={input_info.get('quantization_zero_point', '')}",
        f"- Output shape: {output_info.get('shape', '')}",
        f"- Output dtype: {output_info.get('dtype', '')}",
        f"- Output quantization: scale={output_info.get('quantization_scale', '')}, zero_point={output_info.get('quantization_zero_point', '')}",
        "",
        "## Operators",
        "",
        ", ".join(f"`{op}`" for op in operators) if operators else "Operator list unavailable.",
        "",
        "## Accuracy context",
        "",
        f"- Int8 test accuracy: {format_metric(metrics.get('int8_accuracy'))}",
        f"- Int8 test loss: {format_metric(metrics.get('int8_loss'))}",
        "",
        "## ESP32 integration steps",
        "",
        "1. Create an ESP32 Arduino or PlatformIO project with TensorFlow Lite Micro support.",
        f"2. Copy `{Path(report['model_array_header']).name}` into the sketch folder.",
        f"3. Copy the structure from `{Path(report['example_sketch']).name}`.",
        "4. Replace the zero-filled input sample with a normalized ECG heartbeat window of length 187.",
        "5. Flash the board and check whether `AllocateTensors()` succeeds.",
        "6. If allocation fails, increase `kTensorArenaSize` or reduce the model.",
        "7. Measure serial `inference_us` on the board and record the real ESP32 inference time.",
        "",
        "## Limitations",
        "",
        "- The tensor arena is an estimate. Final memory depends on the exact TFLite Micro version and ESP32 build flags.",
        "- The generated sketch is a minimal inference skeleton, not a full data-acquisition firmware.",
        "- The model is suitable for ESP32 testing because the int8 TFLite artifact is small, but hardware validation still needs to be run on the board.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_metric(value) -> str:
    if value is None or value == "":
        return "not recorded"
    try:
        if pd.isna(value):
            return "not recorded"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
