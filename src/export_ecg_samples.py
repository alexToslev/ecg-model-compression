"""Export reproducible real MIT-BIH ECG samples as int8 C arrays."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse sample-selection, quantization, and output-header options."""
    parser = argparse.ArgumentParser(
        description="Export real MIT-BIH test beats as int8 C arrays for ESP32 TFLite Micro inference."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/processed/mitbih_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/esp32_final_cap_4"))
    parser.add_argument("--report", type=Path, default=Path("results/esp32_final_cap_4/esp32_deployment_report.json"))
    parser.add_argument("--model", type=Path, default=Path("results/final_candidate_cap_4/tiny_ecg_cnn_int8.tflite"))
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible per-class sample selection.",
    )
    parser.add_argument("--prefer-correct", action="store_true")
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--input-zero-point", type=int, default=None)
    parser.add_argument("--model-header", type=str, default="model.h")
    parser.add_argument("--sample-header", type=str, default="ecg_samples.h")
    parser.add_argument("--model-array-name", type=str, default="g_model")
    return parser.parse_args()


def main() -> None:
    """Select samples, quantize them, and write ESP32 helper source files."""
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1")

    input_scale, input_zero_point = resolve_input_quantization(args)
    features, labels, row_indices = select_samples(
        csv_path=args.csv,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        prefer_correct=args.prefer_correct,
        model_path=args.model,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
    )
    quantized = quantize_features(features, input_scale, input_zero_point)

    # The generated header is intentionally self-contained so the ESP32 firmware
    # can run a fixed, reproducible hardware sanity test without loading CSVs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_header_path = args.output_dir / args.sample_header
    main_functions_path = args.output_dir / "main_functions_real_samples.cc"

    write_sample_header(
        output_path=sample_header_path,
        quantized=quantized,
        labels=labels,
        row_indices=row_indices,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
    )
    write_main_functions_example(
        output_path=main_functions_path,
        model_header=args.model_header,
        sample_header=args.sample_header,
        model_array_name=args.model_array_name,
    )

    print(
        json.dumps(
            {
                "sample_header": str(sample_header_path),
                "main_functions_example": str(main_functions_path),
                "sample_count": int(quantized.shape[0]),
                "sample_length": int(quantized.shape[1]),
                "labels": labels.tolist(),
                "csv_row_indices": row_indices.tolist(),
                "prefer_correct": args.prefer_correct,
                "seed": args.seed,
                "input_scale": input_scale,
                "input_zero_point": input_zero_point,
            },
            indent=2,
        )
    )


def resolve_input_quantization(args: argparse.Namespace) -> tuple[float, int]:
    """Resolve input quantization from CLI arguments or a saved report JSON."""
    if args.input_scale is not None and args.input_zero_point is not None:
        return args.input_scale, args.input_zero_point

    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        input_info = report.get("input") or {}
        scale = input_info.get("quantization_scale", args.input_scale)
        zero_point = input_info.get("quantization_zero_point", args.input_zero_point)
        if scale is not None and zero_point is not None:
            return float(scale), int(zero_point)

    missing = []
    if args.input_scale is None:
        missing.append("--input-scale")
    if args.input_zero_point is None:
        missing.append("--input-zero-point")
    raise ValueError(
        f"Could not resolve input quantization. Provide {' and '.join(missing)} "
        f"or a report JSON with input quantization: {args.report}"
    )


def select_samples(
    csv_path: Path,
    samples_per_class: int,
    seed: int,
    prefer_correct: bool,
    model_path: Path,
    input_scale: float,
    input_zero_point: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a reproducible set of test beats, optionally requiring correctness."""
    if not csv_path.exists():
        raise FileNotFoundError(f"MIT-BIH test CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    values = df.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 188:
        raise ValueError(f"Expected 187 ECG values plus one label column, got shape {values.shape}")

    features = values[:, :-1]
    labels = values[:, -1].astype(np.int32)
    selected_indices: list[int] = []
    rng = np.random.default_rng(seed)
    predicted_labels = load_tflite_predictions(model_path, features, input_scale, input_zero_point) if prefer_correct else None
    for label in sorted(np.unique(labels)):
        # Sampling without replacement keeps each exported beat unique while the
        # seed makes the hardware test repeatable.
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size < samples_per_class:
            raise ValueError(f"Class {label} only has {label_indices.size} rows")
        if predicted_labels is not None:
            label_indices = label_indices[predicted_labels[label_indices] == label]
            if label_indices.size < samples_per_class:
                raise ValueError(f"Class {label} only has {label_indices.size} correctly predicted rows")
        selected_indices.extend(
            rng.choice(label_indices, size=samples_per_class, replace=False).tolist()
        )

    rows = np.asarray(selected_indices, dtype=np.int32)
    return features[rows], labels[rows], rows


def load_tflite_predictions(
    model_path: Path,
    features: np.ndarray,
    input_scale: float,
    input_zero_point: int,
) -> np.ndarray:
    """Run the TFLite model over candidate samples for correctness filtering."""
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError("--prefer-correct requires TensorFlow to run the TFLite model") from error

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    predictions = np.empty(features.shape[0], dtype=np.int32)

    for index, feature in enumerate(features):
        # Only argmax is needed here; this helper filters samples, not metrics.
        quantized = quantize_features(feature[np.newaxis, :], input_scale, input_zero_point)
        interpreter.set_tensor(input_info["index"], quantized.reshape(input_info["shape"]))
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])
        predictions[index] = int(np.argmax(output.reshape(-1)))

    return predictions


def quantize_features(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize float ECG feature rows to the model's int8 input range."""
    if scale <= 0.0:
        raise ValueError("Input scale must be positive")
    quantized = np.round(features / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def write_sample_header(
    output_path: Path,
    quantized: np.ndarray,
    labels: np.ndarray,
    row_indices: np.ndarray,
    input_scale: float,
    input_zero_point: int,
) -> None:
    """Write selected ECG samples, labels, and source CSV rows as a C header."""
    sample_count, sample_length = quantized.shape
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "",
        f"constexpr int kEcgSampleCount = {sample_count};",
        f"constexpr int kEcgSampleLength = {sample_length};",
        f"constexpr float kEcgInputScale = {input_scale:.12g}f;",
        f"constexpr int kEcgInputZeroPoint = {input_zero_point};",
        "",
        "const int8_t kEcgSamples[kEcgSampleCount][kEcgSampleLength] = {",
    ]
    for sample in quantized:
        lines.append("  {")
        # Keep 16 values per row so the generated sample header remains readable.
        for start in range(0, sample_length, 16):
            values = ", ".join(f"{int(value):4d}" for value in sample[start : start + 16])
            lines.append(f"    {values},")
        lines.append("  },")
    lines.extend(
        [
            "};",
            "",
            "const int kEcgSampleLabels[kEcgSampleCount] = {",
            "  " + ", ".join(str(int(label)) for label in labels) + ",",
            "};",
            "",
            "const int kEcgSampleCsvRows[kEcgSampleCount] = {",
            "  " + ", ".join(str(int(row)) for row in row_indices) + ",",
            "};",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_main_functions_example(
    output_path: Path,
    model_header: str,
    sample_header: str,
    model_array_name: str,
) -> None:
    """Write a reference TFLite Micro setup/loop using the exported samples."""
    source = f"""
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/micro_interpreter.h"
    #include "tensorflow/lite/micro/system_setup.h"
    #include "tensorflow/lite/schema/schema_generated.h"

    #include "esp_timer.h"
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"

    #include "main_functions.h"
    #include "{model_header}"
    #include "{sample_header}"

    namespace {{
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int sample_index = 0;

    constexpr int kTensorArenaSize = 112 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    }}  // namespace

    void setup() {{
      tflite::InitializeTarget();

      model = tflite::GetModel({model_array_name});
      if (model->version() != TFLITE_SCHEMA_VERSION) {{
        MicroPrintf("Model schema %d does not match supported schema %d",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
      }}

      static tflite::MicroMutableOpResolver<12> resolver;
      resolver.AddConv2D();
      resolver.AddDepthwiseConv2D();
      resolver.AddFullyConnected();
      resolver.AddReshape();
      resolver.AddSoftmax();
      resolver.AddMaxPool2D();
      resolver.AddExpandDims();
      resolver.AddMean();

      static tflite::MicroInterpreter static_interpreter(
          model, resolver, tensor_arena, kTensorArenaSize);
      interpreter = &static_interpreter;

      if (interpreter->AllocateTensors() != kTfLiteOk) {{
        MicroPrintf("AllocateTensors() failed");
        return;
      }}

      input = interpreter->input(0);
      output = interpreter->output(0);

      MicroPrintf("Loaded ECG model with %d real test samples", kEcgSampleCount);
      MicroPrintf("Input bytes=%d scale=%f zero_point=%d", static_cast<int>(input->bytes),
                  static_cast<double>(input->params.scale), input->params.zero_point);
    }}

    void loop() {{
      if (interpreter == nullptr || input == nullptr || output == nullptr) {{
        vTaskDelay(pdMS_TO_TICKS(1000));
        return;
      }}

      if (static_cast<int>(input->bytes) != kEcgSampleLength) {{
        MicroPrintf("Unexpected input size: got %d expected %d",
                    static_cast<int>(input->bytes), kEcgSampleLength);
        vTaskDelay(pdMS_TO_TICKS(2000));
        return;
      }}

      for (int i = 0; i < kEcgSampleLength; ++i) {{
        input->data.int8[i] = kEcgSamples[sample_index][i];
      }}

      int64_t start_us = esp_timer_get_time();
      TfLiteStatus invoke_status = interpreter->Invoke();
      int elapsed_us = static_cast<int>(esp_timer_get_time() - start_us);
      if (invoke_status != kTfLiteOk) {{
        MicroPrintf("Invoke failed");
        vTaskDelay(pdMS_TO_TICKS(2000));
        return;
      }}

      int predicted_class = 0;
      int8_t best_q = output->data.int8[0];
      float best_score = (best_q - output->params.zero_point) * output->params.scale;
      MicroPrintf("Sample %d csv_row=%d true_class=%d",
                  sample_index, kEcgSampleCsvRows[sample_index],
                  kEcgSampleLabels[sample_index]);
      MicroPrintf("ECG class outputs:");
      for (int i = 0; i < 5; ++i) {{
        int8_t q = output->data.int8[i];
        float score = (q - output->params.zero_point) * output->params.scale;
        if (score > best_score) {{
          best_score = score;
          best_q = q;
          predicted_class = i;
        }}
        MicroPrintf("class %d: q=%d, score=%f", i, q, static_cast<double>(score));
      }}

      MicroPrintf("predicted_class=%d true_class=%d best_q=%d best_score=%f inference_us=%d",
                  predicted_class, kEcgSampleLabels[sample_index], best_q,
                  static_cast<double>(best_score), elapsed_us);
      MicroPrintf("----");

      sample_index = (sample_index + 1) % kEcgSampleCount;
      vTaskDelay(pdMS_TO_TICKS(2000));
    }}
    """
    output_path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
