# Structured MLP Pruning Workflow

This document describes WP4. The final WP4 path uses structured hidden-neuron
pruning for the manual MLP. The older unstructured magnitude pruning code is
kept in `src/prune_mlp.py` for optional comparison, but it is not the primary
WP4 result.

## Why Structured Pruning

Magnitude pruning removes individual weights. This can create many zeros, but
the dense layer shapes stay the same unless a later storage or deployment step
uses sparse formats.

Structured pruning removes whole hidden neurons. This is easier to explain and
closer to deployable compression because an inactive neuron can be interpreted
as removed model capacity.

For this project, structured MLP pruning:

- ranks hidden neurons by the norm of their incoming weights
- deactivates a fraction of the weakest hidden neurons
- propagates those inactive hidden outputs into the next layer's input
  connections
- keeps the output class logits intact
- reports both parameter sparsity and hidden-neuron structured sparsity

## Main Command

Run structured pruning on demo data:

```bash
python -m src prune-mlp --demo-data --weights /tmp/ecg_mlp_smoke/baseline_mlp_weights.npz --output-dir /tmp/ecg_mlp_pruning_structured
```

Run structured pruning on the full baseline MLP:

```bash
python -m src prune-mlp --weights results/baseline_mlp/baseline_mlp_weights.npz --data-dir data/processed --output-dir results/baseline_mlp/pruning_structured
```

The default prune fractions are:

```text
0.0,0.2,0.4,0.6,0.8,0.9
```

Override them with:

```bash
python -m src prune-mlp --weights results/baseline_mlp/baseline_mlp_weights.npz --data-dir data/processed --output-dir results/baseline_mlp/pruning_structured --prune-fractions 0.0,0.25,0.5,0.75
```

## Optional Magnitude Pruning

The magnitude pruning implementation is still available for comparison:

```bash
python -m src.prune_mlp --magnitude --weights results/baseline_mlp/baseline_mlp_weights.npz --data-dir data/processed --output-dir results/baseline_mlp/pruning_magnitude
```

Use this only as a secondary comparison. The WP4 report should emphasize the
structured results.

## Output Artifacts

The pruning command writes:

- `pruned_00.npz`, `pruned_20.npz`, ...
- `pruning_metrics.csv`
- `pruning_metrics.json`
- `pruning_summary.md`
- `pruning_sparsity_vs_accuracy.png`
- `pruning_fraction_vs_accuracy.png`
- `pruning_sparsity_vs_loss.png`
- `pruning_structured_sparsity_vs_accuracy.png`

The metrics include:

- pruning method
- prune fraction
- parameter sparsity
- structured hidden-neuron sparsity
- total, zero, and nonzero parameters
- active and pruned hidden-neuron counts
- estimated float32 size from nonzero parameters
- validation and test loss/accuracy

## Interpretation Notes

The saved `.npz` files still store dense arrays, so actual file size may not
shrink in proportion to sparsity. The `estimated_size_bytes` field estimates the
float32 size if only nonzero parameters were stored.

For final reporting, compare the structured pruning table against the baseline
MLP metrics from `results/baseline_mlp/metrics.json`.
