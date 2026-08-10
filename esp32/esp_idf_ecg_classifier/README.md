# ECG Classifier for ESP32

This ESP-IDF firmware runs the final fully INT8 1D CNN for five-class ECG
heartbeat classification with TensorFlow Lite Micro.

## Final configuration

- Target: ESP32 (Xtensa), 160 MHz, 4 MB flash
- ESP-IDF: 5.4.4
- TensorFlow Lite Micro component: 1.3.7
- Model: cap-4 CNN, 11,173 parameters
- Embedded TFLite model: 23,576 bytes
- Tensor arena: 80 KiB
- Input: 187 fully quantized INT8 ECG values
- Output: five INT8 class probabilities

The firmware contains 25 reproducibly selected MIT-BIH test beats: five per
class, sampled without replacement with NumPy seed 42. These samples are a
hardware sanity test, not a replacement for evaluation on the complete test
set.

## Build and run

Open the ESP-IDF PowerShell installed with ESP-IDF 5.4.4 and run:

```powershell
cd "path\to\esp32\esp_idf_ecg_classifier"
idf.py -B C:\ecg-final-build build
idf.py -B C:\ecg-final-build -p COM5 flash monitor
```

Replace `COM5` with the serial port of the connected ESP32. Press `Ctrl+]` to
exit the monitor. After the first build, rebuilding, flashing, and monitoring
can be combined:

```powershell
idf.py -B C:\ecg-final-build -p COM5 build flash monitor
```

## Expected output

For each beat, the firmware prints the CSV row, true class, five quantized
scores, predicted class, correctness, and inference latency. After all 25
beats, it prints total correct predictions plus mean, minimum, and maximum
latency.

## Reference model results

- Full INT8 test accuracy: 95.12%
- Full INT8 macro F1: 0.8055
- Balanced 25-beat hardware check: 21/25 correct (84.00%)
- Balanced 25-beat mean ESP32 latency: 52.09 ms per beat

## Important limitation

The firmware currently uses embedded test beats. It demonstrates on-device
inference but does not yet acquire or preprocess a live ECG sensor signal and
is not a clinically validated system.
