/*
 * utils.h
 */

#ifndef INC_UTILS_H_
#define INC_UTILS_H_

#include <stdint.h>
#include "arm_math.h"

void start_cycle_count();
void stop_cycle_count(char *s);

void hex_encode(char* s, const uint8_t* buf, size_t len);

static inline float q15_to_float(q15_t val) {
    return (float)val / 32768.0f;
}

#endif /* INC_UTILS_H_ */
