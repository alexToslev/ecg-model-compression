#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tiny_ecg_cnn_int8_model.h"

namespace {
constexpr int kTensorArenaSize = 177152;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::AllOpsResolver resolver;
}

void setup() {
  Serial.begin(115200);
  model = tflite::GetModel(g_tiny_ecg_cnn_int8_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed; increase kTensorArenaSize");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("ECG CNN model loaded on ESP32");
}

void loop() {
  if (input == nullptr || output == nullptr) {
    delay(1000);
    return;
  }

  // Replace this zero-filled sample with one normalized MIT-BIH heartbeat of length 187.
  for (int i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = input->params.zero_point;
  }

  unsigned long start_us = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long elapsed_us = micros() - start_us;

  if (status != kTfLiteOk) {
    Serial.println("Inference failed");
    delay(1000);
    return;
  }

  int best_class = 0;
  int8_t best_score = output->data.int8[0];
  for (int i = 1; i < output->bytes; ++i) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_class = i;
    }
  }

  Serial.print("Predicted ECG class: ");
  Serial.print(best_class);
  Serial.print(" inference_us=");
  Serial.println(elapsed_us);
  delay(1000);
}
