# WP1-WP11 Implementation Roadmap

This roadmap reconciles the PDF proposal, `PROJECT_AUDIT.md`, and the current
repository state. It is the working checklist for finishing the project one
small work package at a time.

## Source of Truth

- Branch: `cnn_implementation`
- Proposal file: `Project_Proposal_AP.pdf`
- Audit file: `PROJECT_AUDIT.md`
- Current implementation: manual NumPy MLP and CNN pipelines with compression
  experiments under `src/`

The PDF proposal defines WP1-WP11. The older `project_proposal.md` is useful
background, but it is shorter and should not override the PDF work-package
scope.

## Completion Status

| WP | Proposal scope | Current status | Remaining gap |
|---|---|---:|---|
| WP1 | Literature review, repository setup, development environment | Partial | Add clean setup notes and literature summary |
| WP2 | ECG dataset loading, normalization, train/validation/test split, signal visualization | Mostly complete | Add a clear preprocessing diagnostics artifact |
| WP3 | Baseline MLP implementation and training | Mostly complete | Preserve final run outputs and document the workflow |
| WP4 | Magnitude-based MLP pruning and sparsity-vs-accuracy analysis | Complete enough | Curate final outputs and smoke-test the command |
| WP5 | Fixed-point / 8-bit MLP quantization and comparison | Mostly complete | Integrate final comparison text and outputs |
| WP6 | Baseline 1D CNN implementation, training, and evaluation | Mostly complete | Preserve useful 20-epoch results and validate commands |
| WP7 | CNN pruning and compression evaluation | Partial | Finish structured pruning reporting and fix known issues |
| WP8 | CNN quantization and comparison with baseline/pruned models | Partial | Add post-training summary and CNN QAT workflow |
| WP9 | Benchmark all models for accuracy, parameters, memory, and inference efficiency | Largely missing | Add canonical benchmark script and result table |
| WP10 | Optional ESP32 deployment or simplified simulation | Missing | Add export/simulation report or explicit scope note |
| WP11 | Integration, reproducibility checks, documentation, exam preparation | Partial | Clean docs, reproducibility commands, and final result inventory |

## Canonical Result Folders

Final report evidence should come from these folders once they have been
validated or regenerated:

- `results/dataset_visualizations/`
- `results/baseline_mlp/`
- `results/baseline_mlp/pruning/`
- `results/baseline_mlp/pruning_structured/`
- `results/baseline_mlp/quantization/`
- `results/baseline_cnn/`
- `results/baseline_cnn/pruning/`
- future `results/baseline_cnn/quantization/`
- future `results/benchmarks/`
- future `results/tinyml_simulation/`

Temporary or exploratory folders such as `tmp_*`, `results/cnn_epoch_test*`,
`results/demo_baseline_test`, `results/manual_demo_test*`, `results/pruned`,
and `results/quantized` should not be used as final evidence unless a later
work package explicitly promotes or regenerates their contents.

## Sub-WP Plan

### WP1 - Setup, Literature, and Planning

- WP1.1: add this WP1-WP11 roadmap.
- WP1.2: add environment and reproducibility setup notes.
- WP1.3: add a short literature/background summary for ECG classification,
  CNNs, pruning, quantization, and TinyML.
- WP1.4: update ignore rules for temporary/debug outputs identified by the
  audit.

### WP2 - Dataset Pipeline

- WP2.1: verify loader behavior, normalization modes, and stratified split.
- WP2.2: add a preprocessing diagnostics report artifact.
- WP2.3: document dataset commands and expected outputs.

### WP3 - Baseline MLP

- WP3.1: inspect and preserve baseline MLP training outputs.
- WP3.2: add a fast smoke check for MLP training/evaluation behavior.
- WP3.3: document the final baseline MLP workflow.

### WP4 - MLP Pruning

- WP4.1: verify unstructured magnitude pruning.
- WP4.2: verify structured neuron pruning.
- WP4.3: curate pruning metrics, plots, and summary text.

### WP5 - MLP Quantization

- WP5.1: verify manual 8-bit MLP quantization.
- WP5.2: verify MLP quantization-aware training support.
- WP5.3: document MLP compression comparison.

### WP6 - Baseline CNN

- WP6.1: identify and preserve useful 20-epoch CNN results.
- WP6.2: validate baseline CNN training and evaluation commands.
- WP6.3: document CNN baseline artifacts.

### WP7 - Structured CNN Pruning

- WP7.1: fix the CNN pruning summary path and baseline comparison.
- WP7.2: improve structured filter/neuron pruning metrics.
- WP7.3: save clear pruning reports and plots.
- WP7.4: document estimated size versus actual saved file size.

### WP8 - CNN Quantization

- WP8.1: finalize post-training CNN int8 quantization outputs.
- WP8.2: add quantization-aware training for the CNN.
- WP8.3: compare baseline, pruned, PTQ, and QAT CNN results.

### WP9 - Benchmarking

- WP9.1: add a unified benchmark command.
- WP9.2: measure accuracy, model size, parameter count, sparsity, and
  inference time.
- WP9.3: write a final benchmark CSV and Markdown summary.

### WP10 - TinyML Export or Simulation

- WP10.1: decide the achievable ESP32/TinyML path for this repository.
- WP10.2: add either a simple simulation report or export notes.

### WP11 - Final Integration

- WP11.1: clean duplicate temporary artifacts only after documenting why.
- WP11.2: run reproducibility smoke checks.
- WP11.3: polish README and final technical documentation.
- WP11.4: prepare a final result inventory for the oral exam.

## Development Rule

Work must continue one sub-WP at a time. After each sub-WP, stop and record:

- changed files
- reason for the change
- command to test
- exact suggested git commit command and commit message
