# Results Guide

This directory intentionally preserves the project's experimental progression. Some folders contain exploratory, smoke-test, or superseded results; they are retained for traceability and should not be interpreted as the final reported model.

## Authoritative final results

Use these folders for the final report and poster:

- `final_cap4_comparison/` — authoritative full-test-set comparison between the final float32 cap-4 CNN and the fully INT8 model. The INT8 model achieves **95.12% test accuracy**, **0.8055 macro F1**, and a **23,576-byte** model size, a **73.91% reduction** from float32.
- `esp32_final_cap_4/` — final ESP32 hardware evidence. The balanced 25-beat run achieved **21/25 correct predictions** with **52.090 ms mean inference latency** and an **80 KiB tensor arena**. This small hardware subset validates deployment; the full PC-side INT8 evaluation remains the authoritative accuracy result.
- `presentation_plots/` — presentation-ready visualizations derived from the final results.

## Experiment progression

| Phase | Folder | Purpose and interpretation |
| --- | --- | --- |
| Dataset understanding | `dataset_visualizations/` | Canonical class-distribution and representative-beat plots for the MIT-BIH data used by the project. |
| Initial baseline | `baseline_cnn/` | First Keras 1D-CNN baseline and its evaluation artifacts. Kept as the starting point for comparison. |
| QAT exploration | `baseline_cnn_qat/` | Quantization-aware-training experiment. This was investigated but was not selected as the final deployment path. |
| Early improved model | `improved_cnn/` | Intermediate architecture and imbalance-handling experiments. Not the final reported configuration. |
| NumPy CNN bring-up | `scratch_cnn/` | Early from-scratch implementation checks and small outputs used while validating the NumPy CNN. |
| Smoke validation | `smoke_scratch_cnn/` | Short smoke-test run used to verify that the from-scratch training pipeline executed correctly. Not a performance result. |
| Reduced-data check | `improved_cnn_scratch_small/` | Small-data/short-run diagnostic experiment. Retained for debugging provenance, not for model comparison. |
| Full improved NumPy CNN | `improved_cnn_scratch/` | Full from-scratch CNN experiment and quantization artifacts leading toward the final model. |
| Sampling study | `sampling_sweep_cap_4/` | Controlled cap-4 comparison of natural shuffle, weighted sampling, and balanced mini-batches. Shuffle reached **95.26%** float32 accuracy; weighted and balanced sampling overcorrected the imbalance and reached **16.95%** and **17.49%**, respectively. Natural shuffle was therefore retained. |
| Quantization work | `quantized/` | Earlier quantization exports and checks made before the final cap-4 comparison package. |
| Early ESP32 work | `esp32/` | Prototype/simulation-era ESP32 artifacts. Retained as deployment history; use `esp32_final_cap_4/` for final hardware evidence. |

## Reading the duplicated plots

Some experiment folders contain their own copies of dataset plots or confusion matrices. They are intentionally kept beside the corresponding metrics so each historical run remains self-contained. Folder names identify the experiment phase; the final numbers must always be taken from `final_cap4_comparison/` and `esp32_final_cap_4/`.

## Final model selection

The final model was not declared universally optimal. It was selected as the best tested deployment trade-off under this project's constraints:

- compact 11,173-parameter 1D CNN;
- capped class-weighted loss plus mild morphology-preserving augmentation;
- natural shuffle after weighted and balanced sampling performed poorly;
- fully INT8 TensorFlow Lite conversion with representative-data calibration;
- only 0.14 percentage points of accuracy lost relative to float32 while reducing model size by 73.91%; and
- successful ESP32 execution at approximately 52 ms per beat.

For reproducibility and final numerical values, see the Markdown and CSV summaries inside the two authoritative folders above.
