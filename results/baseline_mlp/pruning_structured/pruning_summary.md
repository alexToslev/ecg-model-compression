# MLP Pruning Evaluation Summary

This report shows the effect of structured hidden-neuron pruning on the manual MLP baseline.

The magnitude-based path is kept in the code for optional comparison, but the WP4 result path is structured pruning.

## Summary metrics

- Pruning method: structured
- Prune fractions: 0.0, 0.2, 0.4, 0.6, 0.8, 0.9
- Parameter sparsity values: 0.0000, 0.2436, 0.4666, 0.6541, 0.8373, 0.9150
- Structured sparsity values: 0.0000, 0.2031, 0.4062, 0.5938, 0.7969, 0.8906

## Best observed results

- Best validation accuracy: 0.9296 at sparsity 0.0000
- Best test accuracy: 0.9381 at sparsity 0.0000

## Plots

- `pruning_sparsity_vs_accuracy.png`
- `pruning_fraction_vs_accuracy.png`
- `pruning_sparsity_vs_loss.png`
- `pruning_structured_sparsity_vs_accuracy.png`

## Full results table

| method | prune_fraction | parameter_sparsity | structured_sparsity | active_hidden_neurons | pruned_hidden_neurons | estimated_size_bytes | val_accuracy | test_accuracy | val_loss | test_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured | 0.00 | 0.0000 | 0.0000 | 128 | 0 | 65536 | 0.9296 | 0.9381 | 0.2854 | 0.2573 |
| structured | 0.20 | 0.2436 | 0.2031 | 102 | 26 | 49572 | 0.9064 | 0.9205 | 0.4330 | 0.3326 |
| structured | 0.40 | 0.4666 | 0.4062 | 76 | 52 | 34960 | 0.7208 | 0.7346 | 0.9962 | 0.9297 |
| structured | 0.60 | 0.6541 | 0.5938 | 52 | 76 | 22672 | 0.7680 | 0.7992 | 1.0726 | 1.0069 |
| structured | 0.80 | 0.8373 | 0.7969 | 26 | 102 | 10660 | 0.6584 | 0.6430 | 1.2631 | 1.2412 |
| structured | 0.90 | 0.9150 | 0.8906 | 14 | 114 | 5572 | 0.8357 | 0.8371 | 1.3930 | 1.3754 |