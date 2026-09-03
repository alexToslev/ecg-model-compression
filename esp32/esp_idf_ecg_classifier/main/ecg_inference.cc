// TensorFlow Lite Micro inference loop for the deployed ECG classifier.

#include "ecg_inference.h"

#include <cstdint>
#include <limits>

#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ecg_samples.h"
#include "model.h"

namespace {
constexpr int kClassCount = 5;
constexpr int kTensorArenaSize = 80 * 1024;

// TFLite Micro requires a statically allocated arena because dynamic heap use is
// avoided on small embedded targets.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int sample_index = 0;
int correct_count = 0;
int64_t total_latency_us = 0;
int64_t minimum_latency_us = std::numeric_limits<int64_t>::max();
int64_t maximum_latency_us = 0;

void ResetRunStatistics() {
  // Reset after each full pass over the embedded sample set.
  correct_count = 0;
  total_latency_us = 0;
  minimum_latency_us = std::numeric_limits<int64_t>::max();
  maximum_latency_us = 0;
}

void PrintRunSummary() {
  // Summarize the balanced 25-beat hardware sanity run.
  const double mean_latency_ms =
      static_cast<double>(total_latency_us) / kEcgSampleCount / 1000.0;

  MicroPrintf("=== Hardware test complete ===");
  MicroPrintf("correct: %d/%d", correct_count, kEcgSampleCount);
  MicroPrintf("accuracy: %.2f%%",
              100.0 * static_cast<double>(correct_count) / kEcgSampleCount);
  MicroPrintf("latency mean: %.3f ms", mean_latency_ms);
  MicroPrintf("latency min: %.3f ms",
              static_cast<double>(minimum_latency_us) / 1000.0);
  MicroPrintf("latency max: %.3f ms",
              static_cast<double>(maximum_latency_us) / 1000.0);
  MicroPrintf("==============================");
}
}  // namespace

void InitializeEcgClassifier() {
  // Set up platform hooks and validate that the embedded flatbuffer matches the
  // schema supported by the linked TFLite Micro runtime.
  tflite::InitializeTarget();

  model = tflite::GetModel(g_ecg_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d does not match supported schema %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  // Register only the operators used by the exported model to keep firmware
  // size and resolver memory smaller than the all-ops resolver.
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
    interpreter = nullptr;
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  // The deployment path is fully INT8; float tensors would hide a broken
  // quantization/export step.
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    MicroPrintf("Expected fully int8 model tensors");
    interpreter = nullptr;
    return;
  }

  sample_index = 0;
  ResetRunStatistics();

  MicroPrintf("ECG classifier ready");
  MicroPrintf("model size: %u bytes", g_ecg_model_size);
  MicroPrintf("test samples: %d (5 per class, seed 42)", kEcgSampleCount);
  MicroPrintf("tensor arena: %d bytes", kTensorArenaSize);
}

void RunNextEcgSample() {
  // One call feeds one embedded heartbeat, invokes the model, and logs the
  // prediction and latency over serial.
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    MicroPrintf("ECG interpreter is not ready");
    return;
  }

  if (static_cast<int>(input->bytes) != kEcgSampleLength) {
    MicroPrintf("Unexpected input size: got %d, expected %d",
                static_cast<int>(input->bytes), kEcgSampleLength);
    return;
  }

  for (int index = 0; index < kEcgSampleLength; ++index) {
    input->data.int8[index] = kEcgSamples[sample_index][index];
  }

  // Measure only Invoke(), not serial printing or sample-copy overhead.
  const int64_t start_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = interpreter->Invoke();
  const int64_t elapsed_us = esp_timer_get_time() - start_us;
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed for sample %d", sample_index);
    return;
  }

  int predicted_class = 0;
  int8_t best_quantized_score = output->data.int8[0];

  MicroPrintf("Sample %d, CSV row %d, true class %d", sample_index,
              kEcgSampleCsvRows[sample_index],
              kEcgSampleLabels[sample_index]);
  MicroPrintf("ECG class outputs:");

  for (int class_index = 0; class_index < kClassCount; ++class_index) {
    const int8_t quantized_score = output->data.int8[class_index];
    // Dequantized scores are printed for readability; class selection uses the
    // quantized values directly because all outputs share one scale/zero point.
    const float score =
        (quantized_score - output->params.zero_point) * output->params.scale;
    if (quantized_score > best_quantized_score) {
      best_quantized_score = quantized_score;
      predicted_class = class_index;
    }
    MicroPrintf("class %d: q=%d, score=%f", class_index, quantized_score,
                static_cast<double>(score));
  }

  const bool is_correct =
      predicted_class == kEcgSampleLabels[sample_index];
  correct_count += is_correct ? 1 : 0;
  total_latency_us += elapsed_us;
  if (elapsed_us < minimum_latency_us) {
    minimum_latency_us = elapsed_us;
  }
  if (elapsed_us > maximum_latency_us) {
    maximum_latency_us = elapsed_us;
  }

  MicroPrintf("predicted class: %d", predicted_class);
  MicroPrintf("correct: %s", is_correct ? "yes" : "no");
  MicroPrintf("inference time: %lld us", elapsed_us);
  MicroPrintf("----");

  ++sample_index;
  if (sample_index == kEcgSampleCount) {
    PrintRunSummary();
    sample_index = 0;
    ResetRunStatistics();
  }
}
