# Structured CNN Pruning Workflow

This document describes WP7: applying pruning techniques to the ECG CNN and
evaluating compression effects. The primary WP7 method is structured pruning.
Unstructured magnitude pruning is kept only as a secondary comparison path.

## Main Goal

The target model is the from-scratch 1D CNN from `src/models/cnn1d.py`.
Structured pruning should deactivate whole model structures where possible:

- convolution filters
- downstream convolution input channels connected to pruned filters
- hidden dense neurons

The output class layer is not structurally pruned, because removing class logits
would change the classifier output definition.

## What the Implementation Does

Structured CNN pruning:

1. Ranks convolution filters by filter weight norm.
2. Deactivates the weakest filters for each convolution layer.
3. Propagates pruned filter masks into the next convolution layer's input
   channels.
4. Propagates the final convolution filter mask into the dense layer input
   features.
5. Deactivates hidden dense neurons and propagates that mask into the output
   layer inputs.
6. Evaluates validation and test loss/accuracy for each pruning fraction.

This is stronger than only setting individual low-magnitude weights to zero,
because it reports filter/channel-level structured sparsity.

## Main Command

Structured pruning is the default:

```bash
python -m src prune-cnn --weights results/baseline_cnn/tiny_ecg_cnn_weights.npz --data-dir data/processed --output-dir results/baseline_cnn/pruning
```

Fast demo-data smoke command:

```bash
python -m src train --demo-data --epochs 1 --output-dir /tmp/ecg_cnn_prune_smoke
python -m src prune-cnn --demo-data --weights /tmp/ecg_cnn_prune_smoke/tiny_ecg_cnn_weights.npz --output-dir /tmp/ecg_cnn_prune_structured --prune-fractions 0.0,0.5,0.9
```

## Optional Magnitude Comparison

Magnitude pruning is available only for comparison:

```bash
python -m src prune-cnn --magnitude --weights results/baseline_cnn/tiny_ecg_cnn_weights.npz --data-dir data/processed --output-dir results/baseline_cnn/pruning_magnitude
```

Do not present this as the main CNN pruning result. The professors asked for
structured CNN pruning.

## Output Artifacts

The structured pruning command writes:

- `pruned_00.npz`, `pruned_20.npz`, ...
- `pruning_metrics.csv`
- `pruning_metrics.json`
- `pruning_summary.md`
- `pruning_sparsity_vs_accuracy.png`
- `pruning_fraction_vs_accuracy.png`
- `pruning_sparsity_vs_estimated_size.png`
- `pruning_size_vs_accuracy.png`
- `pruning_structured_sparsity_vs_accuracy.png`

Important metrics:

- `method`
- `prune_fraction`
- `sparsity`
- `structured_sparsity`
- `active_conv_filters`
- `pruned_conv_filters`
- `active_hidden_neurons`
- `pruned_hidden_neurons`
- `nonzero_parameters`
- `estimated_size_bytes`
- validation and test loss/accuracy

## Size Caveat

`estimated_size_bytes` is not the actual `.npz` file size. It estimates the
float32 size if only nonzero parameters were stored.

The current implementation keeps dense arrays with zeros. Actual deployment
size reduction would require one of these later steps:

- physically rebuilding a smaller CNN architecture
- storing sparse tensors
- exporting to a deployment format that compresses zero parameters

This caveat must be included in the final report.

## Final Report Interpretation

Use structured pruning plots and tables to answer:

- How much accuracy is lost as filters/channels are removed?
- How many convolution filters remain active?
- How much parameter sparsity is introduced?
- What is the estimated compressed size?
- Does pruning preserve enough performance to justify compression?
