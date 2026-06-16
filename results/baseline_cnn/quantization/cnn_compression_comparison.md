# CNN Compression Comparison

This report compares the baseline CNN with available structured-pruned, post-training quantized, and quantization-aware-trained CNN results.

| model | method | test_accuracy | test_loss | size_bytes | parameters | parameter_sparsity | structured_sparsity | epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_cnn | float32 | 0.8753 | 0.3835 | 247144 | 49781 | 0.0000 | 0.0000 | 20 |
| structured_pruned_cnn | structured_pruning | 0.9246 | 0.3816 | 123924 | 30981 | 0.3777 | 0.2167 | 20 |
| ptq_int8_cnn | post_training_quantization | 0.8749 | 0.4586 | 62856 | 49781 | 0.0000 | 0.0000 | 20 |
| qat_cnn | quantization_aware_training | 0.8455 | 0.4754 | 247144 | 49781 | 0.0000 | 0.0000 | 20 |

Generated plots:

- `cnn_compression_accuracy.png`
- `cnn_compression_size.png`
- `cnn_compression_accuracy_vs_size.png`

Size note: pruned CNN size is estimated from nonzero float32 parameters unless the architecture is physically shrunk or sparse storage is used.
