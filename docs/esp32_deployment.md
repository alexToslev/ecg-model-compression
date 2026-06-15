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

These commands write:

- `esp32/ecg_tflite_micro/model_data.h`
- `esp32/ecg_tflite_micro/test_sample.h`
- `esp32/ecg_tflite_micro/test_sample.json`

The ESP32 will not train the model. It will load the already trained int8 model, copy one quantized ECG sample into the input tensor, run inference, and print the predicted class over Serial.

## Board

Initial target board:

- ESP32-WROOM-32
- Arduino IDE
- no PSRAM assumed

Because the current int8 model is small, the first test should fit without PSRAM. The tensor arena size will still need to be adjusted during the Arduino upload/debug step.
