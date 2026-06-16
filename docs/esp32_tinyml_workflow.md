# ESP32 / TinyML deployment workflow

WP10 prepares the quantized CNN for ESP32 deployment and documents the remaining hardware-validation steps. The project does not claim a flashed-board result unless the model is actually run on the device.

## WP10.1 Deployment decision

The deployment target is the post-training int8 CNN exported as TensorFlow Lite:

- model: `results/baseline_cnn/tiny_ecg_cnn_int8.tflite`
- input: one ECG heartbeat segment of length 187
- output: five heartbeat classes
- deployment path: TensorFlow Lite Micro on ESP32

This is the best candidate for hardware because it has the smallest CNN artifact and keeps int8 inference, which is appropriate for resource-constrained microcontrollers.

## Generate the ESP32 export package

Run:

```bash
MPLCONFIGDIR=/tmp/mplconfig python -m src export-tinyml \
  --model results/baseline_cnn/tiny_ecg_cnn_int8.tflite \
  --metrics results/baseline_cnn/int8_metrics.json \
  --benchmark results/benchmarks/model_benchmark_comparison.csv \
  --output-dir results/esp32
```

This writes:

- `results/esp32/tiny_ecg_cnn_int8_model.h`
- `results/esp32/esp32_tflite_micro_example.ino`
- `results/esp32/esp32_deployment_report.json`
- `results/esp32/esp32_deployment_report.md`

The header converts the `.tflite` model into a C byte array suitable for embedding in ESP32 firmware. The Arduino sketch is a minimal TensorFlow Lite Micro inference skeleton.

## WP10.2 Simulation report

The report includes:

- model size in bytes
- estimated tensor arena size
- estimated RAM footprint for tensor arena plus I/O tensors
- input and output tensor shapes and quantization parameters
- operator list from the TFLite interpreter
- int8 accuracy from the saved quantization metrics
- desktop benchmark inference time if WP9 results are available
- a clear decision field explaining whether the export is ready for ESP32 simulation

The tensor arena estimate is intentionally conservative. On real hardware, the final `kTensorArenaSize` should be adjusted until `AllocateTensors()` succeeds with a little safety margin.

## Hardware validation steps

1. Create an ESP32 Arduino or PlatformIO project with TensorFlow Lite Micro support.
2. Copy `tiny_ecg_cnn_int8_model.h` into the sketch folder.
3. Start from `esp32_tflite_micro_example.ino`.
4. Replace the placeholder zero-filled input with a normalized ECG heartbeat window of length 187.
5. Flash the ESP32.
6. Confirm that `AllocateTensors()` succeeds.
7. Read `inference_us` from serial output and compare it to the WP9 desktop benchmark.
8. If allocation fails, increase `kTensorArenaSize`; if RAM is still too high, use a smaller model or stronger structured pruning.

## Limitations

- The generated sketch is not a complete ECG acquisition firmware.
- This WP creates an export and simulation report, not proof of board execution.
- Final memory use depends on the TensorFlow Lite Micro version, ESP32 board variant, compiler flags, and resolver choice.
- The sketch uses `AllOpsResolver` for simplicity. For a smaller firmware, replace it with a `MicroMutableOpResolver` containing only the operators listed in the report.
