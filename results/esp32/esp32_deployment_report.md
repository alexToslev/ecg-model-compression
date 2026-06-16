# ESP32 / TinyML Deployment Simulation Report

## Decision

- Deployment status: `export_ready_for_esp32_simulation`
- Hardware execution was not performed in this WP; this report prepares the model for ESP32/TinyML testing.

## Exported files

- C model header: `results/esp32/tiny_ecg_cnn_int8_model.h`
- Example ESP32 sketch: `results/esp32/esp32_tflite_micro_example.ino`
- Source TFLite model: `results/baseline_cnn/tiny_ecg_cnn_int8.tflite`

## Model footprint

- TFLite model size: 62856 bytes
- Estimated tensor arena: 177152 bytes
- Estimated total RAM for arena plus I/O tensors: 177344 bytes
- Assumed flash budget: 4194304 bytes
- Assumed SRAM budget: 327680 bytes

## Model interface

- Input shape: [1, 187, 1]
- Input dtype: int8
- Input quantization: scale=0.02109804004430771, zero_point=-7
- Output shape: [1, 5]
- Output dtype: int8
- Output quantization: scale=0.00390625, zero_point=-128

## Operators

`CONV_2D`, `EXPAND_DIMS`, `FULLY_CONNECTED`, `MAX_POOL_2D`, `PACK`, `RESHAPE`, `SHAPE`, `SOFTMAX`, `STRIDED_SLICE`

## Accuracy and benchmark context

- Int8 test accuracy: 0.8749
- Int8 test loss: 0.4586
- Benchmark inference time on development machine: 0.0092 ms/sample

## ESP32 integration steps

1. Create an ESP32 Arduino or PlatformIO project with TensorFlow Lite Micro support.
2. Copy `tiny_ecg_cnn_int8_model.h` into the sketch folder.
3. Copy the structure from `esp32_tflite_micro_example.ino`.
4. Replace the zero-filled input sample with a normalized ECG heartbeat window of length 187.
5. Flash the board and check whether `AllocateTensors()` succeeds.
6. If allocation fails, increase `kTensorArenaSize` or reduce the model.
7. Measure serial `inference_us` on the board and compare it against the desktop benchmark.

## Limitations

- The tensor arena is an estimate. Final memory depends on the exact TFLite Micro version and ESP32 build flags.
- The generated sketch is a minimal inference skeleton, not a full data-acquisition firmware.
- The model is suitable for ESP32 testing because the int8 TFLite artifact is small, but hardware validation still needs to be run on the board.
