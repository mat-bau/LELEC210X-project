/*
 * spectrogram.c
 */

#include <stdio.h>
#include "spectrogram.h"
#include "spectrogram_tables.h"
#include "config.h"
#include "utils.h"
#include "arm_absmax_q15.h"

q15_t buf    [  SAMPLES_PER_MELVEC  ];
q15_t buf_fft[2*SAMPLES_PER_MELVEC  ];
q15_t buf_tmp[  SAMPLES_PER_MELVEC/2];

// ========== SPARSE MEL STORAGE ==========
#define MAX_NONZERO_PER_FILTER 60

typedef struct {
    uint16_t index;
    q15_t value;
} SparseMelEntry;

static SparseMelEntry mel_sparse[MELVEC_LENGTH][MAX_NONZERO_PER_FILTER];
static uint8_t mel_nonzero_count[MELVEC_LENGTH] = {0};
static int sparse_initialized = 0;
// ========================================

void Spectrogram_Format(q15_t *buf)
{
    arm_shift_q15(buf, 3, buf, SAMPLES_PER_MELVEC);

    for(uint16_t i=0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] -= (1 << 14);
    }
}

// Initialize sparse Mel matrix
static void init_sparse_mel(void)
{
    if (sparse_initialized) return;

    for (int mel = 0; mel < MELVEC_LENGTH; mel++) {
        int count = 0;
        for (int bin = 0; bin < SAMPLES_PER_MELVEC/2; bin++) {
            q15_t val = hz2mel_mat[mel * (SAMPLES_PER_MELVEC/2) + bin];
            if (val != 0) {
                mel_sparse[mel][count].index = bin;
                mel_sparse[mel][count].value = val;
                count++;
                if (count >= MAX_NONZERO_PER_FILTER) break;
            }
        }
        mel_nonzero_count[mel] = count;
    }
    sparse_initialized = 1;
}

void Spectrogram_Init(void)
{
    init_sparse_mel();
}

void Spectrogram_Compute(q15_t *samples, q15_t *melvec)
{
    // STEP 1: Windowing
    arm_mult_q15(samples, hamming_window, buf, SAMPLES_PER_MELVEC);

    // STEP 2: FFT
    arm_rfft_instance_q15 rfft_inst;
    arm_rfft_init_q15(&rfft_inst, SAMPLES_PER_MELVEC, 0, 1);
    arm_rfft_q15(&rfft_inst, buf, buf_fft);

    // STEP 3.1: Find max
    q15_t vmax;
    uint32_t pIndex = 0;
    arm_absmax_q15(buf_fft, SAMPLES_PER_MELVEC, &vmax, &pIndex);
    if (vmax == 0) vmax = 1;

    // STEP 3.2: Normalize
    for (int i = 0; i < SAMPLES_PER_MELVEC; i++) {
        buf[i] = (q15_t) (((q31_t) buf_fft[i] << 15) / ((q31_t)vmax));
    }

    // STEP 3.3: Complex magnitude
    arm_cmplx_mag_q15(buf, buf, SAMPLES_PER_MELVEC/2);

    // STEP 3.4: Denormalize
    for (int i = 0; i < SAMPLES_PER_MELVEC/2; i++) {
        buf[i] = (q15_t) ((((q31_t) buf[i]) * ((q31_t) vmax)) >> 15);
    }

    // STEP 4: SPARSE Mel transform
    if (!sparse_initialized) init_sparse_mel();

    for (int mel = 0; mel < MELVEC_LENGTH; mel++) {
        q63_t sum = 0;
        for (int k = 0; k < mel_nonzero_count[mel]; k++) {
            uint16_t idx = mel_sparse[mel][k].index;
            q15_t val = mel_sparse[mel][k].value;
            sum += (q63_t)buf[idx] * val;
        }
        const q63_t Q30_MAX = 1073741823LL;
        const q63_t Q30_MIN = -1073741824LL;
        if (sum > Q30_MAX) sum = Q30_MAX;
        if (sum < Q30_MIN) sum = Q30_MIN;
        melvec[mel] = (q15_t)(sum >> 15);
    }
}
