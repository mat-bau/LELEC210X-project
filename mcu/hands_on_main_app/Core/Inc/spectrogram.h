#ifndef INC_SPECTROGRAM_H_
#define INC_SPECTROGRAM_H_

#include "arm_math.h"

void Spectrogram_Init(void);
void Spectrogram_Format(q15_t *buf);
void Spectrogram_Compute(q15_t *samples, q15_t *melvec);

#endif /* INC_SPECTROGRAM_H_ */
