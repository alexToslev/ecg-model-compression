# Firmware component sources and private ESP-IDF dependencies.
idf_component_register(
    SRCS
        "main.cc"
        "ecg_inference.cc"
        "model.cc"
    PRIV_REQUIRES
        esp_timer
    INCLUDE_DIRS
        "."
)
