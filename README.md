# ECG Model Compression for Embedded Arrhythmia Classification

This project trains a compact one-dimensional CNN for MIT-BIH heartbeat
classification, compresses it with full INT8 TensorFlow Lite quantization, and
deploys the final model on an ESP32 with TensorFlow Lite Micro.

The core training implementation is written from scratch with NumPy so the
forward pass, backward pass, class-weighted softmax loss, and optimizer updates
can be inspected directly. TensorFlow/Keras is used after training for model
export and TensorFlow Lite conversion.

## Project Structure

```text
src/                         Python training, evaluation, export, and plotting code
src/models/cnn1d.py          Manual NumPy Conv1D CNN implementation
src/data/                    MIT-BIH CSV loading and dataset visualization helpers
src/compression/             TensorFlow Lite INT8 quantization utilities
src/evaluation/              Plot and summary generation scripts
esp32/esp_idf_ecg_classifier Final ESP-IDF firmware for on-device inference
results/                     Saved experiment summaries and final evidence
deliverables/                Final report, poster, and presentation materials
data/                        Dataset placeholder and instructions
```

For final numerical results, use `results/final_cap4_comparison/` and
`results/esp32_final_cap_4/`. Older result folders are retained only as
experiment history and should not be treated as the final reported model.

## Dataset

The large MIT-BIH CSV files are intentionally not included in Git. Place the
preprocessed heartbeat CSV files here before running training or evaluation:

```text
data/processed/mitbih_train.csv
data/processed/mitbih_test.csv
```

Each row should contain 187 ECG sample values followed by one integer class
label in the final column. See `data/README.md` for the dataset format.

## Installation

Create a Python environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The main Python entry point is:

```bash
python -m src --help
```

## Model Architecture

```text
Input: 187 x 1 heartbeat window
Conv1D: 16 filters, kernel 7
ReLU
MaxPool1D
Conv1D: 32 filters, kernel 5
ReLU
MaxPool1D
Conv1D: 64 filters, kernel 3
ReLU
GlobalAveragePool1D
Dense: 32 units
ReLU
Dropout
Dense: 5 classes
Weighted Softmax Cross-Entropy
```

## Training

Run a quick smoke test without the real dataset:

```bash
python -m src train --demo-data --epochs 1 --output-dir results/smoke_scratch_cnn
```

Run the full from-scratch CNN training pipeline:

```bash
python -m src train --data-dir data/processed --output-dir results/improved_cnn_scratch --epochs 20 --batch-size 128 --optimizer adam --learning-rate 0.001 --class-weights balanced --class-weight-cap 10 --augment-rare-classes --rare-target-count 2000
```

The training command writes metrics, learning curves, confusion matrices, saved
weights, and a Keras export into the selected output directory.

## Quantization

Convert a trained Keras model to a fully INT8 TensorFlow Lite model:

```bash
python -m src quantize --model results/improved_cnn_scratch/tiny_ecg_cnn.keras --data-dir data/processed --output results/improved_cnn_scratch/tiny_ecg_cnn_int8.tflite
```

The quantization step uses representative training samples for calibration and
writes INT8 metrics, a classification report, and a confusion matrix next to the
TFLite file.

## ESP32 Deployment

The final firmware is in:

```text
esp32/esp_idf_ecg_classifier/
```

It embeds the final INT8 model and 25 reproducibly selected MIT-BIH test beats
as C arrays. The embedded beat arrays make the hardware run repeatable without
requiring the ESP32 to read the large CSV dataset or a live ECG sensor.

Build and flash from an ESP-IDF 5.4.4 shell:

```bash
cd esp32/esp_idf_ecg_classifier
idf.py build
idf.py -p COM5 flash monitor
```

Replace `COM5` with the serial port for the connected ESP32 board.

## Final Results

The selected deployment model has 11,173 trainable parameters. The final INT8
model achieves 95.12% test accuracy and 0.8055 macro F1 on the full test set,
with a 23,576-byte TFLite model size. This is a 73.91% size reduction compared
with the float32 Keras model.

The final ESP32 hardware run used 25 embedded test beats, five per class, and
achieved 21/25 correct predictions with a mean inference latency of 52.090 ms
per beat using an 80 KiB tensor arena.

## Clean Submission Notes

For a code-only ZIP, keep the source folders, README files, requirements file,
ESP32 firmware, and final evidence summaries. Exclude generated caches and large
local data:

```text
__pycache__/
*.pyc
data/processed/*.csv
mit-bih-arrhythmia-database-*/
*:Zone.Identifier
```

The historical `results/esp32/` folder is an earlier deployment simulation and
is not the final ESP32 firmware. Use `esp32/esp_idf_ecg_classifier/` for final
code and `results/esp32_final_cap_4/` for final hardware evidence.
