// ESP-IDF entry point that repeatedly runs the embedded ECG sample set.

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ecg_inference.h"

extern "C" void app_main(void) {
  // Initialize once, then replay one quantized heartbeat at a steady interval.
  InitializeEcgClassifier();

  while (true) {
    RunNextEcgSample();
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}
