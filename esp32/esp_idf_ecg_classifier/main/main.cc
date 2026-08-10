#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ecg_inference.h"

extern "C" void app_main(void) {
  InitializeEcgClassifier();

  while (true) {
    RunNextEcgSample();
    vTaskDelay(pdMS_TO_TICKS(250));
  }
}
