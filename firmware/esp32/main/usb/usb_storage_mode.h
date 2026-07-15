#pragma once

#include <stdbool.h>
#include "esp_err.h"

bool usb_storage_should_enter_mode(void);
esp_err_t usb_storage_request_mode(bool enabled);
bool usb_storage_music_ready_on_sd(void);
void usb_storage_prepare_sdcard(void);
esp_err_t usb_storage_mode_run(void);
