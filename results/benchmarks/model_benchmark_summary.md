# Unified Model Benchmark

- Dataset: MIT-BIH processed CSV
- Train samples: 75355
- Validation samples: 13299
- Test samples: 21007
- Timing samples: 256
- Timing repeats: 3
- Normalization: none

## Main comparison

| model | family | method | epochs | accuracy | loss | params | sparsity | structured sparsity | size bytes | ms/sample | dataset |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_mlp | MLP | float32 | 20 | 0.9381 | 0.2573 | 32901 | 0.0000 | 0.0000 | 199514 | 0.0005 | MIT-BIH processed CSV |
| pruned_mlp | MLP | structured_pruning | 20 | 0.9000 | 0.5576 | 13163 | 0.2031 |  | 52652 | 0.0005 | MIT-BIH processed CSV |
| quantized_mlp | MLP | manual_int8 | 20 | 0.9378 | 0.2571 | 32901 | 0.0000 | 0.0000 | 16940 | 0.0099 | MIT-BIH processed CSV |
| baseline_cnn | CNN | float32 | 20 | 0.8753 | 0.3835 | 49781 | 0.0000 | 0.0000 | 247144 | 0.5907 | MIT-BIH processed CSV |
| structured_pruned_cnn | CNN | structured_pruning | 20 | 0.8500 | 0.5589 | 39612 | 0.2043 |  | 158448 | 0.5300 | MIT-BIH processed CSV |
| ptq_int8_cnn | CNN | post_training_int8_tflite | 20 | 0.8749 |  | 49781 | 0.0000 | 0.0000 | 62856 | 0.0125 | MIT-BIH processed CSV |
| qat_cnn | CNN | quantization_aware_training |  |  |  |  |  |  |  |  | MIT-BIH processed CSV |

## Summary interpretation

- Best recorded test accuracy: `baseline_mlp` at 0.9381.
- Smallest recorded model artifact: `quantized_mlp` at 16940 bytes.
- Fastest measured inference path in this run: `pruned_mlp` at 0.0005 ms/sample.
- Pruned model sizes are estimated from remaining nonzero float32 parameters unless a physically smaller architecture or sparse storage is used.
- Missing cells mean the corresponding older artifact did not record that metric, not that the value is zero.
- Timing is measured on this machine and should be reported with the hardware/software context.

## Generated plots

- `benchmark_accuracy.png`
- `benchmark_size.png`
- `benchmark_inference_time.png`
- `benchmark_parameters.png`
- `benchmark_sparsity.png`
- `benchmark_accuracy_size_time.png`
