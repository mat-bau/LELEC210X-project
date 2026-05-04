#include "adc_dblbuf.h"
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"

extern ADC_HandleTypeDef hadc1;
extern TIM_HandleTypeDef htim3;

static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE];
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];

static uint32_t packet_cnt = 0;
static volatile int32_t rem_n_bufs = 0;

static volatile uint8_t listening = 1;
static volatile uint8_t acquiring = 0;

int StartADCAcq(int32_t n_bufs) {
    if (n_bufs > 0) {
        rem_n_bufs = n_bufs;
        cur_melvec = 0;
        acquiring = 1;
        listening = 0;
        return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
    }
    return HAL_OK;
}

int IsADCFinished(void) {
    return (rem_n_bufs == 0 && !acquiring);
}

static void StopADCAcq() {
    HAL_ADC_Stop_DMA(&hadc1);
    acquiring = 0;
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
    start_cycle_count();
    DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
    for(unsigned int j=0; j < N_MELVECS; j++) {
        DEBUG_PRINT("FV #%u:\t", j+1);
        for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
            DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[j][i]));
        }
        DEBUG_PRINT("\r\n");
    }
    stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
    char hex_encoded_packet[2*PACKET_LENGTH+1];
    hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
    DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}

static void encode_packet(uint8_t *packet, uint32_t *pc) {
    for (size_t i=0; i<N_MELVECS; i++) {
        for (size_t j=0; j<MELVEC_LENGTH; j++) {
            (packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2]   = mel_vectors[i][j] >> 8;
            (packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2+1] = mel_vectors[i][j] & 0xFF;
        }
    }
    make_packet(packet, PAYLOAD_LENGTH, 0, *pc);
    *pc += 1;
    if (*pc == 0) {
        DEBUG_PRINT("Packet counter overflow!\r\n");
        Error_Handler();
    }
}

static void send_spectrogram() {
    uint8_t packet[PACKET_LENGTH];

    start_cycle_count();
    encode_packet(packet, &packet_cnt);
    stop_cycle_count("Encode packet");

#if ENABLE_RADIO
    start_cycle_count();
    S2LP_Send(packet, PACKET_LENGTH);
    stop_cycle_count("S2LP Send");
#endif

    print_encoded_packet(packet);
}

void ADC_StartListening(void) {
    if ((htim3.Instance->CR1 & TIM_CR1_CEN) == 0) {
        HAL_TIM_Base_Start(&htim3);
    }
    listening = 1;
    acquiring = 0;
    rem_n_bufs = 0;
    cur_melvec = 0;
    HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance != ADC1) return;
    uint16_t buf_idx = 1;

    if (listening) {
        const uint16_t *raw = (const uint16_t *)ADCData[buf_idx];
        int16_t max_val = 0;
        for (uint32_t i = 0; i < ADC_BUF_SIZE; i++) {
            int16_t sample = (int16_t)raw[i] - 2048;
            if (sample < 0) sample = -sample;
            if (sample > max_val) max_val = sample;
        }
        if (max_val > 200) {
            DEBUG_PRINT("\r\n>>> SOUND level=%d <<<\r\n", max_val);
            listening = 0;
            HAL_ADC_Stop_DMA(&hadc1);
            HAL_Delay(50);
            StartADCAcq(N_MELVECS);
        }
        return;
    }

    if (acquiring) {
        rem_n_bufs--;
        Spectrogram_Format((q15_t *)ADCData[buf_idx]);
        Spectrogram_Compute((q15_t *)ADCData[buf_idx], mel_vectors[cur_melvec]);
        cur_melvec++;

        if (rem_n_bufs == 0) {
            StopADCAcq();
            acquiring = 0;
            print_spectrogram();
            send_spectrogram();
            cur_melvec = 0;
            HAL_Delay(50);
            ADC_StartListening();
        }
    }
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance != ADC1) return;
    uint16_t buf_idx = 0;

    if (listening) {
        const uint16_t *raw = (const uint16_t *)ADCData[buf_idx];
        int16_t max_val = 0;
        for (uint32_t i = 0; i < ADC_BUF_SIZE; i++) {
            int16_t sample = (int16_t)raw[i] - 2048;
            if (sample < 0) sample = -sample;
            if (sample > max_val) max_val = sample;
        }
        if (max_val > 200) {
            DEBUG_PRINT("\r\n>>> SOUND level=%d <<<\r\n", max_val);
            listening = 0;
            HAL_ADC_Stop_DMA(&hadc1);
            HAL_Delay(50);
            StartADCAcq(N_MELVECS);
        }
        return;
    }

    if (acquiring) {
        rem_n_bufs--;
        Spectrogram_Format((q15_t *)ADCData[buf_idx]);
        Spectrogram_Compute((q15_t *)ADCData[buf_idx], mel_vectors[cur_melvec]);
        cur_melvec++;

        if (rem_n_bufs == 0) {
            StopADCAcq();
            acquiring = 0;
            print_spectrogram();
            send_spectrogram();
            cur_melvec = 0;
            HAL_Delay(50);
            ADC_StartListening();
        }
    }
}
