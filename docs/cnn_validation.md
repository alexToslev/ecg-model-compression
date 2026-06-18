# CNN Validation

Use this checklist after changing the improved CNN.

## Smoke Test

```bash
python -m src train --demo-data --epochs 1 --output-dir results/smoke_improved_cnn
```

## Full Training

```bash
python -m src train \
  --data-dir data/processed \
  --epochs 30 \
  --output-dir results/improved_cnn \
  --seed 42 \
  --normalize none
```

## Quantization Check

```bash
python -m src quantize \
  --model results/improved_cnn/tiny_ecg_cnn.keras \
  --data-dir data/processed \
  --output results/improved_cnn/tiny_ecg_cnn_int8.tflite
```

## Files To Inspect

```bash
python -m json.tool results/improved_cnn/metrics.json
python -m json.tool results/improved_cnn/classification_report.json
```

Focus on macro F1 and rare-class recall, not only total accuracy.
