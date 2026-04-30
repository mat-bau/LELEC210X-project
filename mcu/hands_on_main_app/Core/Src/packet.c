/*
 * packet.c
 */

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include "aes.h"
#include <string.h>

extern CRYP_HandleTypeDef hcryp;

const uint8_t AES_Key[16] = {
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

static void swap_bytes_for_hw(uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i += 4) {
        uint8_t tmp0 = data[i];
        uint8_t tmp1 = data[i+1];
        uint8_t tmp2 = data[i+2];
        uint8_t tmp3 = data[i+3];
        data[i]   = tmp3;
        data[i+1] = tmp2;
        data[i+2] = tmp1;
        data[i+3] = tmp0;
    }
}

void tag_cbc_mac_HW(uint8_t *tag, const uint8_t *msg, size_t msg_len)
{
    uint32_t statew[4] = {0};
    uint8_t *cbc_state = (uint8_t*)statew;
    size_t offset = 0;

    hcryp.Init.pKey = (uint8_t *)AES_Key;
    hcryp.Init.KeySize = CRYP_KEYSIZE_128B;
    HAL_CRYP_Init(&hcryp);

    while (offset < msg_len) {
        uint8_t block[16] = {0};
        size_t bytes_to_copy = msg_len - offset;
        if (bytes_to_copy > 16) bytes_to_copy = 16;

        memcpy(block, msg + offset, bytes_to_copy);

        for (int i = 0; i < 16; i++) {
            cbc_state[i] ^= block[i];
        }

        uint8_t temp_state[16];
        memcpy(temp_state, cbc_state, 16);
        swap_bytes_for_hw(temp_state, 16);

        HAL_CRYP_AESECB_Encrypt(&hcryp, temp_state, 16, temp_state, HAL_MAX_DELAY);

        swap_bytes_for_hw(temp_state, 16);
        memcpy(cbc_state, temp_state, 16);

        offset += bytes_to_copy;
    }

    memcpy(tag, cbc_state, 16);
}

int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial)
{
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;

    packet[0] = 0x00;
    packet[1] = sender_id;
    packet[2] = (uint8_t)((payload_len >> 8) & 0xFF);
    packet[3] = (uint8_t)(payload_len & 0xFF);
    packet[4] = (uint8_t)((serial >> 24) & 0xFF);
    packet[5] = (uint8_t)((serial >> 16) & 0xFF);
    packet[6] = (uint8_t)((serial >> 8) & 0xFF);
    packet[7] = (uint8_t)(serial & 0xFF);

    uint8_t hw_tag[16];
    tag_cbc_mac_HW(hw_tag, packet, payload_len + PACKET_HEADER_LENGTH);
    memcpy(packet + payload_len + PACKET_HEADER_LENGTH, hw_tag, 16);

    return packet_len;
}
