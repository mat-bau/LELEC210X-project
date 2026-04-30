/*
 * packet.h
 */

#ifndef INC_PACKET_H_
#define INC_PACKET_H_

#include <stdint.h>
#include <stdlib.h>
#include "config.h"

int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial);

#endif /* INC_PACKET_H_ */
