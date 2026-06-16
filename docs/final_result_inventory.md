# Final Result Inventory and Full-Run Checklist

This file is the WP11 handoff checklist. Use it after committing the cleanup work and before starting the final 20-epoch experiments.

## Clean Repository Policy

The final branch should keep source code and documentation tracked, while generated outputs remain ignored unless selected final evidence is force-added.

Removed during WP11 cleanup:

- ad hoc benchmark scripts: `bench_cnn_batch.py`, `bench_cnn_predict.py`
- temporary pipeline scripts: `continue_pipeline.sh`, `run_full_pipeline.sh`
- chat/debug notes and logs: `chat_summary.txt`, `cnn_train.log`, `pipeline_run.log`
- Python caches: `__pycache__/`
- temporary experiment folders: `tmp_*`
- stale exploratory results: `results/cnn_epoch_test*`, `results/demo_baseline_test`, `results/manual_demo_test*`, `results/pruned`, `results/quantized`, `results/dataset_visualizations`

Preserved locally for safety until the final rerun:

- `results/baseline_mlp/`
- `results/baseline_cnn/`

These contain useful previous 20-epoch evidence and can be overwritten by the final run.

## Final Canonical Output Folders

After the full run, final evidence should be organized as:

| folder | purpose |
|---|---|
| `results/dataset_visualizations/` | dataset class/split/signal plots |
| `results/baseline_mlp/` | final baseline MLP metrics, plots, weights |
| `results/baseline_mlp/pruning_structured/` | final structured-pruned MLP evaluation |
| `results/baseline_mlp/quantization/` | final quantized MLP comparison |
| `results/baseline_cnn/` | final baseline CNN metrics, plots, weights, Keras export |
| `results/baseline_cnn/pruning/` | final structured-pruned CNN evaluation |
| `results/baseline_cnn/quantization/` | final CNN compression comparison |
| `results/baseline_cnn_qat/` | final quantization-aware-trained CNN |
| `results/benchmarks/` | final all-model benchmark CSV, summary, plots |
| `results/esp32/` | final ESP32/TinyML export and simulation report |

## Full Run Order

Run these only after the smoke checks pass.

```bash
python -m src visualize --data-dir data/processed --output-dir results/dataset_visualizations

python -m src train-mlp --data-dir data/processed --epochs 20 --output-dir results/baseline_mlp

python -m src prune-mlp \
  --weights results/baseline_mlp/baseline_mlp_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_mlp/pruning_structured

python -m src quantize-mlp \
  --weights results/baseline_mlp/baseline_mlp_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_mlp/quantization

python -m src train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 128 \
  --output-dir results/baseline_cnn

python -m src prune-cnn \
  --weights results/baseline_cnn/tiny_ecg_cnn_weights.npz \
  --data-dir data/processed \
  --output-dir results/baseline_cnn/pruning

python -m src quantize \
  --model results/baseline_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed \
  --output results/baseline_cnn/tiny_ecg_cnn_int8.tflite

python -m src train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 128 \
  --quantize-aware \
  --output-dir results/baseline_cnn_qat

python -m src compare-cnn \
  --baseline-dir results/baseline_cnn \
  --pruning-dir results/baseline_cnn/pruning \
  --quantized-dir results/baseline_cnn \
  --qat-dir results/baseline_cnn_qat \
  --output-dir results/baseline_cnn/quantization

MPLCONFIGDIR=/tmp/mplconfig python -m src benchmark \
  --data-dir data/processed \
  --output-dir results/benchmarks \
  --timing-samples 512 \
  --repeats 5

MPLCONFIGDIR=/tmp/mplconfig python -m src export-tinyml \
  --model results/baseline_cnn/tiny_ecg_cnn_int8.tflite \
  --metrics results/baseline_cnn/int8_metrics.json \
  --benchmark results/benchmarks/model_benchmark_comparison.csv \
  --output-dir results/esp32
```

## Reproducibility Smoke Checks

Before the full run:

```bash
python -m compileall src
python -m src train-mlp --demo-data --epochs 1 --output-dir /tmp/ecg_smoke_mlp
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_smoke_cnn
python -m src train --demo-data --epochs 1 --quantize-aware --output-dir /tmp/ecg_smoke_cnn_qat
```

Optional extra checks:

```bash
python -m src prune-mlp --demo-data --weights /tmp/ecg_smoke_mlp/baseline_mlp_weights.npz --output-dir /tmp/ecg_smoke_mlp_pruning
python -m src prune-cnn --demo-data --weights /tmp/ecg_smoke_cnn/tiny_ecg_cnn_weights.npz --output-dir /tmp/ecg_smoke_cnn_pruning
python -m src quantize-mlp --demo-data --weights /tmp/ecg_smoke_mlp/baseline_mlp_weights.npz --output-dir /tmp/ecg_smoke_mlp_quant
```

## Exam Explanation Checklist

Be ready to explain:

- why MLP is used first as a simpler validation baseline
- why the CNN is the target model for ECG morphology
- how Conv1D, ReLU, MaxPool1D, Dense, and SoftmaxCrossEntropy are implemented from scratch
- difference between unstructured magnitude pruning and structured filter/channel pruning
- why CNN pruning size is estimated unless the architecture is physically shrunk
- difference between post-training quantization and quantization-aware training
- why TFLite int8 is the deployment candidate for ESP32
- what the unified benchmark measures and why timing depends on hardware
- why the ESP32 report is export/simulation rather than a flashed-board result

## Final Commit of Results

Because `results/` is ignored, add final selected evidence explicitly:

```bash
git add -f results/dataset_visualizations results/baseline_mlp results/baseline_cnn results/baseline_cnn_qat results/benchmarks results/esp32
git commit -m "WP11.4: add final regenerated result inventory"
```
