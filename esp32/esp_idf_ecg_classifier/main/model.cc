// Binds the generated model header to stable names used by inference code.

#include "model.h"

#include "ecg_cnn_cap4_int8_model.h"

const unsigned char* const g_ecg_model_data =
    g_final_ecg_cnn_cap_4_int8_model;
const unsigned int g_ecg_model_size =
    g_final_ecg_cnn_cap_4_int8_model_len;
