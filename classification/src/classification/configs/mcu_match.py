"""
MCU-aligned configs: DSP parameters exactly match the STM32 firmware.

Variants for the N_MEL sweep (20 / 24 / 28 / 32):
    MCUMatchConfig  (N_MEL=20)  Current firmware — no zoom needed at inference.
    Mel24MCUConfig  (N_MEL=24)  Requires MELVEC_LENGTH=24 + new recordings.
    Mel28MCUConfig  (N_MEL=28)  Requires MELVEC_LENGTH=28 + new recordings.
    Mel32MCUConfig  (N_MEL=32)  Requires MELVEC_LENGTH=32 + new recordings.

DSP alignment with MCU firmware (config.h):
    SAMPLE_RATE  = 10200   (ADC rate, no Python resampling needed)
    N_FFT        = 512     (SAMPLES_PER_MELVEC)
    HOP_LENGTH   = 512     (no frame overlap, identical to MCU)
    DURATION_MS  = 1557    → compute_n_frames() = int(1557/1000*10200/512)+1 = 32
    N_MEL        = 20      (MELVEC_LENGTH — changes per variant)

Lime packet-size budget (uint16 values per packet):
    N_MEL=20:  20*32*2 =  1280 bytes  (current, upsampling=8 OK)
    N_MEL=24:  24*32*2 =  1536 bytes
    N_MEL=28:  28*32*2 =  1792 bytes
    N_MEL=32:  32*32*2 =  2048 bytes  (check Lime upsampling budget)

With these parameters, scipy.ndimage.zoom factors at inference = [1.0, 1.0]:
    freq_zoom = cfg.N_MEL / MELVEC_LENGTH = 1.0
    time_zoom = cfg.compute_n_frames() / N_MELVECS = 32/32 = 1.0
→ Zero interpolation artifacts, zero domain gap on the spatial dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base_config import BaseConfig


@dataclass
class MCUMatchConfig(BaseConfig):
    """
    Exact MCU firmware alignment — current default firmware (N_MEL=20, N_MELVECS=32).

    Model input shape: (20, 32, 1) = exact packet shape from the MCU.
    No scipy.ndimage.zoom needed at inference → no interpolation artifacts.
    """
    # -- DSP aligned with MCU firmware (config.h) ----------------------------
    SAMPLE_RATE: int = 10200   # ADC_FREQUENCY in firmware
    N_MEL: int       = 20     # MELVEC_LENGTH = 20
    N_FFT: int       = 512     # SAMPLES_PER_MELVEC = 512
    HOP_LENGTH: int  = 512     # no frame overlap (same as MCU)
    # int(1557/1000 * 10200/512) + 1 = int(31.007) + 1 = 32  ✓
    DURATION_MS: int = 1557

    ENABLE_PITCH_SHIFT: bool  = True
    ENABLE_TIME_STRETCH: bool = True

    # Smaller model head (input is ~10× smaller than BaseConfig 64×87)
    DROPOUT_RATE: float        = 0.25
    COSINE_RESTART_EPOCHS: int = 30

    # Use MCU-matched linear magnitude mel (no log) — matches spectrogram.c DSP output.
    # At inference the MCU Q1.15 int16 values are divided by 32768 → same [0,1] range.
    NORM_MODE: str = "mcu_linear"

    MODEL_DIR: str = "./data/models/mcu_match/run"


@dataclass
class Mel24MCUConfig(MCUMatchConfig):
    """
    MCU-aligned config for MELVEC_LENGTH=24.

    Requires:
      1. Recompile MCU firmware with MELVEC_LENGTH=24 in config.h.
      2. Re-record all classes (the mel computation changes).
      3. Verify Lime packet size (1536 bytes) is within RF budget.

    Model input: (24, 32, 1)
    """
    N_MEL: int     = 24
    MODEL_DIR: str = "./data/models/mel24_mcu/run"


@dataclass
class Mel28MCUConfig(MCUMatchConfig):
    """
    MCU-aligned config for MELVEC_LENGTH=28.

    Requires firmware MELVEC_LENGTH=28 + new recordings.
    Packet payload: 28×32×2 = 1792 bytes.

    Model input: (28, 32, 1)
    """
    N_MEL: int     = 28
    MODEL_DIR: str = "./data/models/mel28_mcu/run"


@dataclass
class Mel32MCUConfig(MCUMatchConfig):
    """
    MCU-aligned config for MELVEC_LENGTH=32.

    Requires firmware MELVEC_LENGTH=32 + new recordings.
    Packet payload: 32×32×2 = 2048 bytes.
    NOTE: verify Lime upsampling budget — larger packets may require
    reducing the upsampling factor (8→4 or 8→2) which degrades RF quality.

    Model input: (32, 32, 1)
    """
    N_MEL: int     = 32
    MODEL_DIR: str = "./data/models/mel32_mcu/run"
