# ESP32 Deployment Notes

This branch prepares the trained int8 TensorFlow Lite model for ESP32 inference.

## Step 1: Generate C Headers

Train and quantize the model on the PC first:

```bash
C:\venvs\ecg\Scripts\python.exe -m src.train_cnn --data-dir data/processed --epochs 20
C:\venvs\ecg\Scripts\python.exe -m src.compression.quantize_tflite --data-dir data/processed
```

Export the quantized `.tflite` model as a C header:

```bash
C:\venvs\ecg\Scripts\python.exe -m src.deployment.export_tflite_to_c
```

Export one quantized ECG test sample:

```bash
C:\venvs\ecg\Scripts\python.exe -m src.deployment.export_test_sample --data-dir data/processed --sample-index 0
```

For a stronger ESP32 test, export one sample from each available heartbeat class:

```bash
C:\venvs\ecg\Scripts\python.exe -m src.deployment.export_test_sample --data-dir data/processed --samples-per-class 1
```

These commands write:

- `esp32/ecg_tflite_micro/model_data.h`
- `esp32/ecg_tflite_micro/test_sample.h`
- `esp32/ecg_tflite_micro/test_sample.json`

The ESP32 will not train the model. It will load the already trained int8 model, copy quantized ECG samples into the input tensor, run inference, and print the predicted class over Serial.

## Step 2: Arduino IDE Sketch

Open this sketch in Arduino IDE:

```text
esp32/ecg_tflite_micro/ecg_tflite_micro.ino
```

The sketch includes:

- `model_data.h` - the int8 TensorFlow Lite model as a C array
- `test_sample.h` - one or more quantized ECG heartbeats and their expected labels

Initial Arduino IDE settings:

- Board: ESP32 Dev Module or the matching ESP32-WROOM-32 board entry
- Port: the serial port shown when the board is connected by USB
- Serial Monitor baud rate: `115200`

You also need an Arduino TensorFlow Lite Micro library that provides headers such as:

```text
tensorflow/lite/micro/micro_interpreter.h
tensorflow/lite/micro/all_ops_resolver.h
```

If Arduino IDE reports that these files are missing, install a TensorFlow Lite Micro / TensorFlowLite_ESP32 compatible library from the Arduino Library Manager or from its GitHub ZIP package, then compile again.

The sketch prints:

- model size
- expected labels
- predicted labels
- raw int8 output scores
- dequantized output scores
- inference time in microseconds
- number of correct samples
- free heap before and after inference

## Board

Initial target board:

- ESP32-WROOM-32
- Arduino IDE
- no PSRAM assumed

Because the current int8 model is small, the first test should fit without PSRAM. The tensor arena size will still need to be adjusted during the Arduino upload/debug step.
