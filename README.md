# ECG Model Compression

From-scratch ECG classification and model compression for resource-constrained hardware.

The project compares:

- baseline MLP
- structured-pruned MLP
- quantized MLP
- baseline 1D CNN
- structured-pruned 1D CNN
- post-training int8 CNN
- quantization-aware-trained CNN
- unified benchmark and ESP32/TinyML export simulation

The CNN and MLP training code is implemented manually with NumPy layers. Keras/TensorFlow Lite is used only for export and deployment-oriented quantization.

## Dataset

Use the preprocessed MIT-BIH heartbeat CSV format:

```text
data/processed/mitbih_train.csv
data/processed/mitbih_test.csv
```

Each row contains one heartbeat segment and the class label in the final column. `data/processed/` is intentionally ignored by Git.

## Setup

```bash
pip install -r requirements.txt
python -m src --help
```

For quick checks without the real dataset, add `--demo-data` to supported commands.

## Final Full Pipeline

Run these commands after the repository cleanup is committed. These commands regenerate the final report-ready artifacts from scratch.

### 1. Dataset Visualizations

```bash
python -m src visualize --data-dir data/processed --output-dir results/dataset_visualizations
```

### 2. Baseline MLP

```bash
python -m src train-mlp --data-dir data/processed --epochs 20 --output-dir results/baseline_mlp
```

### 3. Structured-Pruned MLP

```bash
python -m src prune-mlp \
  --weights results/baseline_mlp/baseline_mlp_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_mlp/pruning_structured
```

### 4. Quantized MLP

```bash
python -m src quantize-mlp \
  --weights results/baseline_mlp/baseline_mlp_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_mlp/quantization
```

### 5. Baseline CNN

```bash
python -m src train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 128 \
  --output-dir results/baseline_cnn
```

### 6. Structured-Pruned CNN

```bash
python -m src prune-cnn \
  --weights results/baseline_cnn/tiny_ecg_cnn_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_cnn/pruning
```

### 7. Post-Training Quantized CNN

```bash
python -m src quantize \
  --model results/baseline_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed \
  --output results/baseline_cnn/tiny_ecg_cnn_int8.tflite
```

### 8. Quantization-Aware-Trained CNN

```bash
python -m src train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 128 \
  --quantize-aware \
  --output-dir results/baseline_cnn_qat
```

### 9. CNN Compression Comparison

```bash
python -m src compare-cnn \
  --baseline-dir results/baseline_cnn \
  --pruning-dir results/baseline_cnn/pruning \
  --quantized-dir results/baseline_cnn \
  --qat-dir results/baseline_cnn_qat \
  --output-dir results/baseline_cnn/quantization
```

### 10. Unified Benchmark

```bash
MPLCONFIGDIR=/tmp/mplconfig python -m src benchmark \
  --data-dir data/processed \
  --output-dir results/benchmarks \
  --timing-samples 512 \
  --repeats 5
```

### 11. ESP32 / TinyML Export Simulation

```bash
MPLCONFIGDIR=/tmp/mplconfig python -m src export-tinyml \
  --model results/baseline_cnn/tiny_ecg_cnn_int8.tflite \
  --metrics results/baseline_cnn/int8_metrics.json \
  --benchmark results/benchmarks/model_benchmark_comparison.csv \
  --output-dir results/esp32
```

## Smoke Checks

Use these before long runs:

```bash
python -m compileall src
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_smoke_mlp
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_smoke_cnn
python -m src train --demo-data --epochs 1 --quantize-aware --output-dir /tmp/ecg_smoke_cnn_qat
```

## Key Documentation

- `docs/work_packages.md`
- `docs/setup_reproducibility.md`
- `docs/dataset_workflow.md`
- `docs/mlp_workflow.md`
- `docs/mlp_pruning_workflow.md`
- `docs/mlp_quantization_workflow.md`
- `docs/cnn_baseline_workflow.md`
- `docs/cnn_pruning_workflow.md`
- `docs/cnn_quantization_workflow.md`
- `docs/unified_benchmark_workflow.md`
- `docs/esp32_tinyml_workflow.md`
- `docs/final_result_inventory.md`

## Result Policy

Generated datasets and experiment outputs are ignored by Git. Commit only selected final evidence with `git add -f results/...` when needed. Temporary/debug runs should go under `/tmp/...` or `tmp_*`.
