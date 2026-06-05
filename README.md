# ecg-model-compression
From-scratch compression of neural networks for ECG classification using pruning, quantization, and ESP32 deployment.

## First baseline: tiny 1D CNN

This first version uses the common preprocessed MIT-BIH heartbeat CSV format:

- `data/processed/mitbih_train.csv`
- `data/processed/mitbih_test.csv`

Each row should contain one heartbeat segment, with the class label in the final column.
The popular Kaggle "Heartbeat Categorization Dataset" provides this format.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a quick smoke test with synthetic ECG-like data:

```bash
python -m src.train_cnn --demo-data --epochs 3
python -m src.compression.quantize_tflite --demo-data
```

Train on the real MIT-BIH CSV files with the larger baseline model:

```bash
python -m src.train_cnn --data-dir data/processed --epochs 20
```

This baseline is intentionally larger than the demo model and is meant to support structured pruning and post-training INT8 quantization. The model architecture is now roughly 300k parameters and follows a stronger 1D CNN design:

- `Conv1D(64)` → `Conv1D(128)` → `MaxPool`
- `Conv1D(256)` → `Conv1D(192)` → `MaxPool`
- `GlobalAveragePooling1D` → `Dense(128)` → output

If your real dataset is provided as a zip archive, put the archive in the repository root and extract it into `data/processed` with:

```bash
python -m src.data.extract_mitbih --zip-file your_dataset.zip
```

If you already have the raw MIT-BIH arrhythmia files in a folder like `mit-bih-arrhythmia-database-1.0.0`, convert them to the required CSV format with:

```bash
python -m src.data.convert_mitbih_wfdb --input-dir mit-bih-arrhythmia-database-1.0.0 --output-dir data/processed
```

This will write:

- `data/processed/mitbih_train.csv`
- `data/processed/mitbih_test.csv`

Convert the trained model to int8 TensorFlow Lite:

```bash
python -m src.compression.quantize_tflite --data-dir data/processed
```

The training script saves:

- `results/baseline_cnn/baseline_ecg_cnn.keras` — full Keras model
- `results/baseline_cnn/history.csv` — epoch-by-epoch loss/accuracy
- `results/baseline_cnn/training_history.png` — training and validation curves
- `results/baseline_cnn/metrics.json` — final test metrics

Outputs are written under `results/`.
