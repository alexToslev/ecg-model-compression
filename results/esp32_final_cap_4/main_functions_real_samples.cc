// Reference TFLite Micro loop used to replay exported real ECG test samples.

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "main_functions.h"
#include "final_ecg_cnn_cap_4_int8_model.h"
#include "ecg_samples.h"

namespace {
// Static interpreter state mirrors the ESP32 firmware structure used for the
// final hardware validation.
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int sample_index = 0;

constexpr int kTensorArenaSize = 112 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  // Initialize the runtime, check model compatibility, and allocate tensors once.
  tflite::InitializeTarget();

  model = tflite::GetModel(g_final_ecg_cnn_cap_4_int8_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d does not match supported schema %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<12> resolver;
  // Register the operators present in the exported ECG CNN graph.
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddMaxPool2D();
  resolver.AddExpandDims();
  resolver.AddMean();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Loaded ECG model with %d real test samples", kEcgSampleCount);
  MicroPrintf("Input bytes=%d scale=%f zero_point=%d", static_cast<int>(input->bytes),
              static_cast<double>(input->params.scale), input->params.zero_point);
}

void loop() {
  // Each loop invocation copies one selected ECG beat, invokes the model, and
  // prints the prediction details used in the hardware run notes.
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    return;
  }

  if (static_cast<int>(input->bytes) != kEcgSampleLength) {
    MicroPrintf("Unexpected input size: got %d expected %d",
                static_cast<int>(input->bytes), kEcgSampleLength);
    vTaskDelay(pdMS_TO_TICKS(2000));
    return;
  }

  for (int i = 0; i < kEcgSampleLength; ++i) {
    input->data.int8[i] = kEcgSamples[sample_index][i];
  }

  // Timing covers only inference, matching the latency reported for ESP32.
  int64_t start_us = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int elapsed_us = static_cast<int>(esp_timer_get_time() - start_us);
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    vTaskDelay(pdMS_TO_TICKS(2000));
    return;
  }

  int predicted_class = 0;
  int8_t best_q = output->data.int8[0];
  float best_score = (best_q - output->params.zero_point) * output->params.scale;
  MicroPrintf("Sample %d csv_row=%d true_class=%d",
              sample_index, kEcgSampleCsvRows[sample_index],
              kEcgSampleLabels[sample_index]);
  MicroPrintf("ECG class outputs:");
  for (int i = 0; i < 5; ++i) {
    int8_t q = output->data.int8[i];
    // Convert quantized logits/scores to float only for serial readability.
    float score = (q - output->params.zero_point) * output->params.scale;
    if (score > best_score) {
      best_score = score;
      best_q = q;
      predicted_class = i;
    }
    MicroPrintf("class %d: q=%d, score=%f", i, q, static_cast<double>(score));
  }

  MicroPrintf("predicted_class=%d true_class=%d best_q=%d best_score=%f inference_us=%d",
              predicted_class, kEcgSampleLabels[sample_index], best_q,
              static_cast<double>(best_score), elapsed_us);
  MicroPrintf("----");

  sample_index = (sample_index + 1) % kEcgSampleCount;
  vTaskDelay(pdMS_TO_TICKS(2000));
}
