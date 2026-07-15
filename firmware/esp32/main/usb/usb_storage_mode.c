#include "usb_storage_mode.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs.h"
#include "sdmmc_cmd.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tinyusb.h"
#include "tusb_cdc_acm.h"
#include "tusb_msc_storage.h"

#include "aura_config.h"
#include "audio/music_library.h"

static const char *TAG = "usb_storage";
static const char *USB_BASE_PATH = "/usb";
static const char *MUSIC_DIR = "/sdcard/music";
static const char *USB_MUSIC_DIR = "/usb/MUSIC";
static const char *USB_NAMESPACE = "system";
static const char *USB_FLAG_KEY = "usb_storage";

static esp_err_t storage_init_sdmmc_card(sdmmc_card_t **out_card)
{
    ESP_RETURN_ON_FALSE(out_card != NULL, ESP_ERR_INVALID_ARG, TAG, "null card output");

    gpio_config_t d3_cfg = {
        .pin_bit_mask = BIT64(SDMMC_D3_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&d3_cfg));
    gpio_set_level(SDMMC_D3_PIN, 1);

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = SDMMC_CLK_PIN;
    slot.cmd = SDMMC_CMD_PIN;
    slot.d0 = SDMMC_DATA_PIN;
    slot.d3 = SDMMC_D3_PIN;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    sdmmc_card_t *card = calloc(1, sizeof(sdmmc_card_t));
    ESP_RETURN_ON_FALSE(card != NULL, ESP_ERR_NO_MEM, TAG, "no memory for sd card");

    esp_err_t ret = host.init();
    if (ret != ESP_OK) {
        free(card);
        return ret;
    }

    ret = sdmmc_host_init_slot(host.slot, &slot);
    if (ret != ESP_OK) {
        if (host.flags & SDMMC_HOST_FLAG_DEINIT_ARG) {
            host.deinit_p(host.slot);
        } else {
            host.deinit();
        }
        free(card);
        return ret;
    }

    ret = sdmmc_card_init(&host, card);
    if (ret != ESP_OK) {
        if (host.flags & SDMMC_HOST_FLAG_DEINIT_ARG) {
            host.deinit_p(host.slot);
        } else {
            host.deinit();
        }
        free(card);
        return ret;
    }

    gpio_reset_pin(SDMMC_D3_PIN);
    *out_card = card;
    return ESP_OK;
}

static void ensure_music_dir(const char *base_path)
{
    char path[64];
    snprintf(path, sizeof(path), "%s/MUSIC", base_path);
    struct stat st = {0};
    if (stat(path, &st) == 0) {
        return;
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST) {
        ESP_LOGW(TAG, "Failed creating %s: %s", path, strerror(errno));
    }
}

static void write_readme_if_missing(const char *base_path)
{
    char path[64];
    snprintf(path, sizeof(path), "%s/README.TXT", base_path);
    struct stat st = {0};
    if (stat(path, &st) == 0) {
        return;
    }

    FILE *f = fopen(path, "w");
    if (!f) {
        ESP_LOGW(TAG, "Failed creating %s", path);
        return;
    }
    fputs("Put MP3 or WAV files into /music and then reboot the device.\n", f);
    fputs("When /music contains audio files, Aura will boot back into normal mode.\n", f);
    fclose(f);
}

bool usb_storage_should_enter_mode(void)
{
    nvs_handle_t nvs = 0;
    uint8_t enabled = 0;
    if (nvs_open(USB_NAMESPACE, NVS_READONLY, &nvs) != ESP_OK) {
        return false;
    }
    esp_err_t err = nvs_get_u8(nvs, USB_FLAG_KEY, &enabled);
    nvs_close(nvs);
    return err == ESP_OK && enabled == 1;
}

esp_err_t usb_storage_request_mode(bool enabled)
{
    nvs_handle_t nvs = 0;
    ESP_RETURN_ON_ERROR(nvs_open(USB_NAMESPACE, NVS_READWRITE, &nvs), TAG, "open usb_storage nvs");
    ESP_RETURN_ON_ERROR(nvs_set_u8(nvs, USB_FLAG_KEY, enabled ? 1 : 0), TAG, "set usb_storage flag");
    esp_err_t ret = nvs_commit(nvs);
    nvs_close(nvs);
    return ret;
}

bool usb_storage_music_ready_on_sd(void)
{
    return music_library_has_supported_files(MUSIC_DIR);
}

void usb_storage_prepare_sdcard(void)
{
    ensure_music_dir("/sdcard");
    write_readme_if_missing("/sdcard");
}

esp_err_t usb_storage_mode_run(void)
{
    usb_storage_request_mode(false);

    sdmmc_card_t *card = NULL;
    ESP_ERROR_CHECK(storage_init_sdmmc_card(&card));

    const tinyusb_msc_sdmmc_config_t config_sdmmc = {
        .card = card,
        .mount_config.max_files = 8,
    };
    ESP_ERROR_CHECK(tinyusb_msc_storage_init_sdmmc(&config_sdmmc));

    const tinyusb_config_t tusb_cfg = {
        .device_descriptor = NULL,
        .string_descriptor = NULL,
        .string_descriptor_count = 0,
        .external_phy = false,
#if (TUD_OPT_HIGH_SPEED)
        .fs_configuration_descriptor = NULL,
        .hs_configuration_descriptor = NULL,
        .qualifier_descriptor = NULL,
#else
        .configuration_descriptor = NULL,
#endif
    };
    ESP_ERROR_CHECK(tinyusb_driver_install(&tusb_cfg));

#if CONFIG_TINYUSB_CDC_ENABLED
    tinyusb_config_cdcacm_t acm_cfg = {
        .usb_dev = TINYUSB_USBDEV_0,
        .cdc_port = TINYUSB_CDC_ACM_0,
        .rx_unread_buf_sz = 64,
    };
    ESP_ERROR_CHECK(tusb_cdc_acm_init(&acm_cfg));
#endif

    ESP_LOGW(TAG, "USB storage mode active. Copy files, eject, then reboot to return to Aura.");
    while (1) {
        if (!tinyusb_msc_storage_in_use_by_usb_host()) {
            esp_err_t ret = tinyusb_msc_storage_mount(USB_BASE_PATH);
            if (ret == ESP_OK) {
                bool ready = music_library_has_supported_files(USB_MUSIC_DIR);
                tinyusb_msc_storage_unmount();
                if (ready) {
                    ESP_LOGW(TAG, "Detected music files after USB eject, rebooting to normal mode");
                    usb_storage_request_mode(false);
                    esp_restart();
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
