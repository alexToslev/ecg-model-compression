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

Run a quick smoke test with synthetic ECG-like data for development:

```bash
python -m src train --demo-data --epochs 3
python -m src quantize --demo-data
```

When the full dataset is available, run using the `data/processed` folder:

```bash
python -m src train --data-dir data/processed --epochs 20
python -m src evaluate --model results/baseline_cnn/tiny_ecg_cnn.keras --data-dir data/processed
python -m src quantize --model results/baseline_cnn/tiny_ecg_cnn.keras --data-dir data/processed
python -m src summarize --run-dir results/baseline_cnn
```

Train on the real MIT-BIH CSV files:

```bash
python -m src.train_cnn --data-dir data/processed --epochs 20
```

Convert the trained model to int8 TensorFlow Lite:

```bash
python -m src.compression.quantize_tflite --data-dir data/processed
```

Outputs are written under `results/`.
