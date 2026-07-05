# Final Cap-4 Baseline vs INT8 Comparison

## Summary table

| Metric                    | Baseline float32 | Quantized INT8 |
| ------------------------- | ---------------- | -------------- |
| Test accuracy             | 0.9526           | 0.9512         |
| Test loss                 | 0.1950           | 0.2145         |
| Macro precision           | 0.7775           | 0.7779         |
| Macro recall              | 0.8555           | 0.8514         |
| Macro F1                  | 0.8050           | 0.8055         |
| Weighted F1               | 0.9536           | 0.9521         |
| Trainable parameters      | 11173            | 11173          |
| Model size bytes          | 90361            | 23576          |
| Size reduction percent    | 0.0000           | 73.9091        |
| Tensor arena KiB          | -                | 80.0000        |
| ESP32 latency ms per beat | -                | 52.0000        |

## Generated quantized plots

- `quantized_plots/int8_class_metrics.png`
- `quantized_plots/int8_confusion_matrix.png`
- `quantized_plots/int8_confusion_matrix_normalized.png`

## Generated comparison plots

- `comparison_plots/baseline_vs_int8_accuracy_loss.png`
- `comparison_plots/baseline_vs_int8_macro_metrics.png`
- `comparison_plots/baseline_vs_int8_per_class_f1.png`
- `comparison_plots/baseline_vs_int8_model_size.png`
- `comparison_plots/int8_deployment_parameters_memory_latency.png`
- `comparison_plots/baseline_vs_int8_confusion_matrices.png`

## Deployment notes

- ESP32 latency is recorded as approximately 52.0 ms per ECG beat from the hardware serial output.
- Tensor arena is recorded as 80.0 KiB from the deployed firmware configuration.
- Full test-set accuracy is measured on PC; ESP32 execution validates that the quantized model runs on hardware with real representative MIT-BIH beats.
