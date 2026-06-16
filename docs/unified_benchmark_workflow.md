# Unified benchmark workflow

WP9 collects the project results into one benchmark table and a small set of readable plots. The benchmark compares the available MLP and CNN artifacts on accuracy, loss, parameter count, sparsity, size, and inference time.

## Models included

The default benchmark looks for:

- baseline MLP: `results/baseline_mlp/`
- structured-pruned MLP: `results/baseline_mlp/pruning_structured/`
- quantized MLP: `results/baseline_mlp/quantization/`
- baseline CNN: `results/baseline_cnn/`
- structured-pruned CNN: `results/baseline_cnn/pruning/`
- post-training int8 CNN: `results/baseline_cnn/int8_metrics.json`
- quantization-aware-trained CNN: `results/baseline_cnn_qat/`

If QAT CNN artifacts are not present yet, the report keeps a row for `qat_cnn` and marks it as missing instead of inventing results.

## Run the benchmark

After the WP1-WP8 artifacts are available, run:

```bash
MPLCONFIGDIR=/tmp/mplconfig python -m src benchmark \
  --data-dir data/processed \
  --output-dir results/benchmarks \
  --timing-samples 512 \
  --repeats 5
```

The command loads the processed MIT-BIH data for inference timing and reads saved metrics for the official accuracy/loss values.

## Outputs

The benchmark writes:

- `results/benchmarks/model_benchmark_comparison.csv`
- `results/benchmarks/model_benchmark_summary.md`
- `results/benchmarks/benchmark_accuracy.png`
- `results/benchmarks/benchmark_size.png`
- `results/benchmarks/benchmark_inference_time.png`
- `results/benchmarks/benchmark_parameters.png`
- `results/benchmarks/benchmark_sparsity.png`
- `results/benchmarks/benchmark_accuracy_size_time.png`

The CSV is the machine-readable result table. The Markdown file is the final human-readable summary and points to the plots.

## Interpretation notes

- Accuracy and loss come from saved test-set evaluation artifacts, not from the smaller timing subset.
- Inference time is measured on the current machine, so report it with the machine/software context.
- Structured-pruned CNN and MLP sizes are estimated from remaining nonzero float32 parameters unless a physically smaller architecture or sparse storage format is exported.
- Post-training CNN quantization reports the actual `.tflite` file size.
- Older artifacts may not contain every metric. Missing cells in the table mean the result was not recorded in that artifact.

## Fast smoke test

For a quick pipeline check:

```bash
MPLCONFIGDIR=/tmp/mplconfig python -m src benchmark \
  --demo-data \
  --timing-samples 64 \
  --repeats 1 \
  --output-dir /tmp/ecg_wp9_benchmark
```

The smoke test checks that the report and plots are generated, but it should not be used as the final project benchmark because it uses synthetic data.
