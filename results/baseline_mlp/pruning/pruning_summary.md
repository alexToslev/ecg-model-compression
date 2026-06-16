# Pruning Evaluation Summary

This report shows the effect of magnitude-based pruning on the manual MLP baseline.

## Summary metrics

- Prune fractions: 0.0, 0.2, 0.4, 0.6, 0.8, 0.9
- Sparsity values: 0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 0.9000

## Best observed results

- Best validation accuracy: 0.9301 at sparsity 0.2000
- Best test accuracy: 0.9385 at sparsity 0.2000

## Plots

- `pruning_sparsity_vs_accuracy.png`
- `pruning_fraction_vs_accuracy.png`
- `pruning_sparsity_vs_loss.png`

## Full results table

| prune_fraction | sparsity | val_accuracy | test_accuracy | val_loss | test_loss |
|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.9296 | 0.9381 | 0.2854 | 0.2573 |
| 0.20 | 0.2000 | 0.9301 | 0.9385 | 0.2862 | 0.2555 |
| 0.40 | 0.4000 | 0.9259 | 0.9346 | 0.3046 | 0.2614 |
| 0.60 | 0.6000 | 0.9004 | 0.9266 | 0.3671 | 0.3111 |
| 0.80 | 0.8000 | 0.8657 | 0.9194 | 0.5054 | 0.3756 |
| 0.90 | 0.9000 | 0.8339 | 0.9258 | 0.7319 | 0.5169 |