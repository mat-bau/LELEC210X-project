/*
 * config.h
 */

#ifndef INC_CONFIG_H_
#define INC_CONFIG_H_

#include <stdio.h>

// Runtime parameters
#define MAIN_APP 0
#define EVAL_RADIO 1

#define RUN_CONFIG MAIN_APP

// Radio parameters
#define ENABLE_RADIO 1

// General UART enable/disable (disable for low-power operation)
#define ENABLE_UART 1

// In continuous mode, we start and stop continuous acquisition on button press.
// In non-continuous mode, we send a single packet on button press.
#define CONTINUOUS_ACQ 0

// Spectrogram parameters
#define SAMPLES_PER_MELVEC 512
#define MELVEC_LENGTH 20
#define N_MELVECS 20

// Enable performance measurements
#define PERF_COUNT 1

// Enable debug print
#define DEBUGP 1

#if (DEBUGP == 1)
#define DEBUG_PRINT(...) do{ printf(__VA_ARGS__ ); } while( 0 )
#else
#define DEBUG_PRINT(...) do{ } while ( 0 )
#endif

#define ADC_BUF_SIZE SAMPLES_PER_MELVEC

#define PACKET_HEADER_LENGTH 8
#define PACKET_TAG_LENGTH 16
#define PAYLOAD_LENGTH (N_MELVECS * MELVEC_LENGTH * 2)
#define PACKET_LENGTH (PACKET_HEADER_LENGTH + PAYLOAD_LENGTH + PACKET_TAG_LENGTH)


#endif /* INC_CONFIG_H_ */
