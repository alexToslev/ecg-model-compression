# Literature and Background Summary

This short summary supports WP1 of the project proposal. It explains why the
project studies ECG classification, 1D CNNs, pruning, quantization, and
TinyML-style deployment constraints.

## Project Motivation

ECG classification is useful for detecting abnormal heart rhythms from
heartbeat signals. A neural network can learn patterns in the signal shape, but
embedded and wearable devices have limited memory, compute power, and energy.
The central project question is therefore:

Can a small ECG classifier keep useful accuracy after compression methods such
as pruning and 8-bit quantization?

The repository answers this by comparing baseline and compressed models using:

- test accuracy and loss
- model size
- parameter count
- sparsity
- inference time

## ECG Classification and MIT-BIH

The MIT-BIH Arrhythmia Database is a common ECG benchmark for heartbeat rhythm
classification. The project uses the preprocessed heartbeat CSV format, where
each row contains 187 ECG samples followed by a class label.

For this project, the ECG pipeline must make the data preparation explicit:

- load train and test CSV files
- split the train file into train and validation sets
- support reproducible random seeds
- support normalization modes
- visualize class balance and signal examples

This matters because compression results are only meaningful if every model is
evaluated on the same data split and preprocessing settings.

## CNNs for ECG Signals

Convolutional neural networks were popularized by work such as LeCun et al. on
gradient-based learning and LeNet-style architectures. A 1D CNN applies the
same idea to time-series signals: small convolution filters scan across the ECG
waveform and learn local patterns.

For ECG classification, a 1D CNN is a good fit because:

- ECG signals are ordered time-series samples.
- Local waveform shapes can indicate heartbeat classes.
- Convolution layers reuse filter weights across the signal.
- Pooling layers reduce sequence length and computation.

This repository implements a small from-scratch 1D CNN with Conv1D, ReLU,
MaxPool1D, Flatten, and Dense layers so the training and compression logic can
be explained clearly in an oral exam.

## Pruning

Magnitude-based pruning is motivated by work such as Han et al., which showed
that many neural-network weights can be removed with limited accuracy loss.
The basic idea is simple: parameters or structures with small magnitude often
contribute less to the output.

The project uses two pruning ideas:

- Unstructured pruning removes individual small weights.
- Structured pruning removes or deactivates larger units such as neurons,
  filters, or channels.

For MLPs, unstructured and structured neuron pruning are both useful. For CNNs,
the professors specifically suggested structured pruning, so the CNN path
should focus on filter-level or channel-level pruning rather than only
individual weight sparsity.

The final report must be careful with terminology:

- If weights are zeroed but the architecture shape is unchanged, the saved
  dense model file may not become smaller automatically.
- Estimated compressed size can be reported from nonzero parameters.
- Actual file size reduction requires sparse storage, architecture shrinking,
  or deployment-specific compression.

## Quantization

Integer quantization is motivated by work such as Jacob et al. on efficient
integer-only neural-network inference. Quantization stores weights and/or
activations using fewer bits, commonly int8 instead of float32.

The main benefit is that int8 values require less memory:

- float32 weight: 4 bytes
- int8 weight: 1 byte

This can reduce model size and improve deployment feasibility on embedded
hardware. The trade-off is that rounding and clipping can reduce accuracy.

The project includes:

- manual fixed-point style MLP quantization
- TensorFlow Lite post-training quantization for exported Keras models
- planned CNN quantization-aware training, where quantization effects are
  simulated during training

Post-training quantization is simpler because it happens after training.
Quantization-aware training is more involved but can preserve accuracy better
because the model learns while exposed to quantization noise.

## Efficient Inference and TinyML

Efficient neural-network computation is a recurring theme in systems such as
cuDNN, which provides optimized deep-learning primitives, and MLPerf Tiny,
which benchmarks machine-learning workloads on very small devices.

This project does not need to implement low-level kernels. Instead, it connects
to the same motivation by reporting practical efficiency metrics:

- parameter count
- estimated memory usage
- model file size
- inference time
- optional ESP32 or simplified TinyML simulation notes

These metrics make the project more than an accuracy experiment. They show the
trade-off between classification performance and resource usage.

## How This Literature Maps to the Work Packages

| Topic | Project work package |
|---|---|
| ECG data and MIT-BIH | WP2 |
| Baseline MLP validation model | WP3 |
| Baseline 1D CNN | WP6 |
| Magnitude and structured pruning | WP4, WP7 |
| Fixed-point and int8 quantization | WP5, WP8 |
| Embedded/TinyML constraints | WP9, WP10 |
| Final explanation and reproducibility | WP11 |

## Exam Explanation

The project can be explained as a compression study:

1. Build reproducible ECG data loading.
2. Train baseline classifiers.
3. Apply pruning to remove less important parameters or structures.
4. Apply quantization to reduce numerical precision.
5. Compare accuracy, size, sparsity, parameters, and inference time.
6. Discuss whether the compressed models are more suitable for constrained
   hardware.

The important technical distinction is that pruning and quantization optimize
different things. Pruning removes model capacity or creates sparsity, while
quantization changes how numbers are represented. A good final comparison
should show both accuracy impact and resource impact.
