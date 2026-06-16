# CNN Pruning Evaluation Summary

This report documents structured filter/channel and hidden-neuron pruning for the scratch 1D CNN.

The WP7 target is structured pruning. Magnitude pruning is available only as a secondary comparison.

Structured pruning deactivates whole convolution filters, propagates pruned channels into downstream convolution inputs, deactivates hidden dense neurons, and keeps output class logits intact.

Pruning here stores dense arrays with zeros. It does not physically rebuild a smaller CNN architecture.

## Summary metrics

- Pruning method: structured
- Prune fractions: 0.0, 0.2, 0.4, 0.6, 0.8, 0.9
- Parameter sparsity values: 0.0000, 0.3777, 0.6456, 0.8332, 0.9539, 0.9852
- Structured sparsity values: 0.0000, 0.2167, 0.4083, 0.6000, 0.7917, 0.8833

## Best observed results

- Best validation accuracy: 0.9429 at sparsity 0.0000
- Best test accuracy: 0.9246 at sparsity 0.3777

## Plots

- `pruning_sparsity_vs_accuracy.png`
- `pruning_fraction_vs_accuracy.png`
- `pruning_sparsity_vs_estimated_size.png`
- `pruning_size_vs_accuracy.png`
- `pruning_structured_sparsity_vs_accuracy.png`

## Full results table

| method | prune_fraction | parameter_sparsity | structured_sparsity | active_conv_filters | pruned_conv_filters | active_hidden_neurons | pruned_hidden_neurons | nonzero_parameters | estimated_size_bytes | val_accuracy | test_accuracy | val_loss | test_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured | 0.00 | 0.0000 | 0.0000 | 56 | 0 | 64 | 0 | 49780 | 199120 | 0.9429 | 0.8753 | 0.2087 | 0.3835 |
| structured | 0.20 | 0.3777 | 0.2167 | 43 | 13 | 51 | 13 | 30981 | 123924 | 0.9053 | 0.9246 | 0.3780 | 0.3816 |
| structured | 0.40 | 0.6456 | 0.4083 | 33 | 23 | 38 | 26 | 17644 | 70576 | 0.8234 | 0.8311 | 0.6832 | 0.6837 |
| structured | 0.60 | 0.8332 | 0.6000 | 22 | 34 | 26 | 38 | 8302 | 33208 | 0.0905 | 0.0741 | 2.1845 | 2.4516 |
| structured | 0.80 | 0.9539 | 0.7917 | 12 | 44 | 13 | 51 | 2295 | 9180 | 0.1102 | 0.1398 | 1.5781 | 1.5574 |
| structured | 0.90 | 0.9852 | 0.8833 | 7 | 49 | 7 | 57 | 739 | 2956 | 0.5700 | 0.3901 | 1.5555 | 1.5533 |

## Original baseline model

- Baseline test accuracy: 0.8753
- Baseline model size: 247144 bytes

The pruning results below compare pruned models against this original baseline model.
Estimated compressed size is derived from the number of nonzero float32 parameters, not from an actually smaller saved model file.


## Size interpretation

`estimated_size_bytes` assumes only nonzero float32 parameters are stored. The `.npz` checkpoint still contains dense arrays with zeros, so actual file size reduction requires sparse storage, architecture shrinking, or a deployment-specific compression format.
