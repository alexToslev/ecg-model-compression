# ESP32 First ECG Inference Result

This note records the first successful TensorFlow Lite Micro inference test on the ESP32-WROOM-32.

## Setup

- Board: ESP32-WROOM-32
- Arduino board selection: ESP32 Dev Module
- USB driver: Silicon Labs CP210x USB to UART Bridge
- Model: full int8 TensorFlow Lite ECG CNN
- Input: one hardcoded ECG heartbeat test sample
- The ESP32 performs inference only. It does not train the model.

## Serial Monitor Output

```text
ets Jul 29 2019 12:21:46

rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)
configsip: 0, SPIWP:0xee
clk_drv:0x00,q_drv:0x00,d_drv:0x00,cs0_drv:0x00,hd_drv:0x00,wp_drv:0x00
mode:DIO, clock div:1
load:0x3fff0030,len:1344
load:0x40078000,len:13964
load:0x40080400,len:3600
entry 0x400805f0

ECG TensorFlow Lite Micro test
Model size bytes: 11928
Sample index: 0
Expected label: 0
Free heap before allocation: 283680
Input bytes=187 type=9 scale=0.00392157 zero_point=-128
Output bytes=5 type=9 scale=0.00390625 zero_point=-128
Predicted label: 0
Expected label: 0
Inference time microseconds: 99751
Free heap after inference: 283480
Output scores:
  class 0: raw=127 dequantized=0.996094
  class 1: raw=-127 dequantized=0.003906
  class 2: raw=-128 dequantized=0.000000
  class 3: raw=-128 dequantized=0.000000
  class 4: raw=-128 dequantized=0.000000
```

## How To Read The Result

The sample is one ECG heartbeat segment from the test set. In this project, each sample contains 187 ECG signal values and one class label.

- `Model size bytes: 11928` means the quantized TFLite model is about 11.9 KB.
- `Input bytes=187` means the model receives 187 int8 values for one heartbeat.
- `Output bytes=5` means the model returns one score for each of the five heartbeat classes.
- `Predicted label: 0` is the class selected by the ESP32 model.
- `Expected label: 0` is the correct label from the dataset.
- `Inference time microseconds: 99751` means the inference took about 99.8 ms.
- `Free heap after inference: 283480` shows that enough RAM remains after running the model.

The ESP32 prediction is correct for this sample because the predicted label and expected label are both `0`.

## Heartbeat Class Meaning

The five labels follow the common AAMI-style heartbeat grouping used for MIT-BIH heartbeat classification:

| Label | Short name | Meaning |
| --- | --- | --- |
| 0 | N | Normal beat |
| 1 | S | Supraventricular ectopic beat |
| 2 | V | Ventricular ectopic beat |
| 3 | F | Fusion beat |
| 4 | Q | Unknown or unclassifiable beat |

In this test, sample `0` is a normal heartbeat. The ESP32 model predicted class `0`, so it also classified the heartbeat as normal.

## Output Scores

The output scores are quantized int8 values. The sketch also converts them back to approximate floating-point confidence values using the output scale and zero point.

For this sample:

- Class 0 has confidence about 0.996.
- Class 1 has confidence about 0.004.
- Classes 2, 3, and 4 have confidence about 0.

This means the model was very confident that the sample belongs to class `0`, normal beat.

## Conclusion

This confirms that the quantized ECG CNN can be loaded and executed on the ESP32-WROOM-32. The first hardcoded test heartbeat was classified correctly, and the model fits comfortably in flash and RAM.

This is not yet a live ECG sensor test. The next step is to run more exported test samples, then later connect real ECG input hardware if required.
