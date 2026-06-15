#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>

#include "model_data.h"
#include "test_sample.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

const tflite::Model* model = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Start generous for ESP32-WROOM-32. If allocation fails, increase this value.
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int argmax_int8(const TfLiteTensor* tensor) {
  int best_index = 0;
  int8_t best_value = tensor->data.int8[0];

  for (int index = 1; index < tensor->bytes; ++index) {
    const int8_t value = tensor->data.int8[index];
    if (value > best_value) {
      best_value = value;
      best_index = index;
    }
  }
  return best_index;
}

float dequantize(int8_t value, float scale, int zero_point) {
  return scale * (static_cast<int>(value) - zero_point);
}

void print_tensor_info(const char* name, const TfLiteTensor* tensor) {
  Serial.print(name);
  Serial.print(" bytes=");
  Serial.print(tensor->bytes);
  Serial.print(" type=");
  Serial.print(tensor->type);
  Serial.print(" scale=");
  Serial.print(tensor->params.scale, 8);
  Serial.print(" zero_point=");
  Serial.println(tensor->params.zero_point);
}

void print_output_scores() {
  Serial.println("Output scores:");
  for (int index = 0; index < output->bytes; ++index) {
    const int8_t raw = output->data.int8[index];
    const float score = dequantize(raw, output->params.scale, output->params.zero_point);
    Serial.print("  class ");
    Serial.print(index);
    Serial.print(": raw=");
    Serial.print(raw);
    Serial.print(" dequantized=");
    Serial.println(score, 6);
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("ECG TensorFlow Lite Micro test");
  Serial.print("Model size bytes: ");
  Serial.println(g_ecg_model_len);
  Serial.print("Sample index: ");
  Serial.println(g_ecg_sample_index);
  Serial.print("Expected label: ");
  Serial.println(g_ecg_expected_label);
  Serial.print("Free heap before allocation: ");
  Serial.println(ESP.getFreeHeap());

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_ecg_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch.");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed. Try increasing kTensorArenaSize.");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  print_tensor_info("Input", input);
  print_tensor_info("Output", output);

  if (input->type != kTfLiteInt8) {
    Serial.println("Expected int8 input tensor.");
    return;
  }
  if (output->type != kTfLiteInt8) {
    Serial.println("Expected int8 output tensor.");
    return;
  }
  if (input->bytes != g_ecg_sample_len) {
    Serial.print("Input length mismatch. Tensor bytes=");
    Serial.print(input->bytes);
    Serial.print(" sample length=");
    Serial.println(g_ecg_sample_len);
    return;
  }

  memcpy(input->data.int8, g_ecg_sample, g_ecg_sample_len);

  const unsigned long start_us = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed.");
    return;
  }
  const unsigned long elapsed_us = micros() - start_us;

  const int predicted_label = argmax_int8(output);
  Serial.print("Predicted label: ");
  Serial.println(predicted_label);
  Serial.print("Expected label: ");
  Serial.println(g_ecg_expected_label);
  Serial.print("Inference time microseconds: ");
  Serial.println(elapsed_us);
  Serial.print("Free heap after inference: ");
  Serial.println(ESP.getFreeHeap());
  print_output_scores();
}

void loop() {
  delay(1000);
}
