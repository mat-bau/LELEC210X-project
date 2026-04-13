"""
Base configuration dataclass for all ResNet audio classifiers.

All parameters live here so that nothing is a mutable global.
Subclass this to create experiment-specific configs (e.g. ResNet32Config).

NOTE: n_frames is computed from Python DSP params ONLY.
      The MCU firmware (config.h) is NEVER touched for this.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../LELEC210X-project/
_DEFAULT_REAL_DATA_DIR = str(_REPO_ROOT / "mcu" / "hands_on_audio_acquisition" / "audio_files")


@dataclass
class BaseConfig:
    # -- DSP -----------------------------------------------------------------
    SAMPLE_RATE: int = 11025    # Hz — must match resampling target in inference
    N_MEL: int = 64             # mel bands
    N_FFT: int = 512            # FFT window size
    HOP_LENGTH: int = 128       # FFT hop  (rule: N_FFT // 4)
    DURATION_MS: int = 1000     # analysis window duration in ms

    # -- Augmentation --------------------------------------------------------
    ENABLE_TIME_SHIFT: bool = True
    ENABLE_PITCH_SHIFT: bool = True
    ENABLE_TIME_STRETCH: bool = True
    ENABLE_NOISE: bool = True
    ENABLE_SPEC_MASKING: bool = True

    PITCH_SHIFT_SEMITONES: tuple = (-3, 3)
    TIME_STRETCH_RANGE: tuple = (0.85, 1.15)
    NOISE_SIGMA: float = 0.04

    # -- Architecture --------------------------------------------------------
    DROPOUT_RATE: float = 0.3
    L2_REG: float = 1e-4
    LABEL_SMOOTHING: float = 0.15
    MIXUP_ALPHA: float = 0.4
    ACTIVATION: str = "relu"    # relu | gelu | swish | mish | leaky_relu

    # -- Training ------------------------------------------------------------
    BATCH_SIZE: int = 256
    EPOCHS: int = 300
    LEARNING_RATE: float = 5e-4
    EARLY_STOPPING_PATIENCE: int = 60
    TTA_STEPS: int = 5

    # -- LR schedule ---------------------------------------------------------
    COSINE_RESTART_EPOCHS: int = 45
    COSINE_M_MUL: float = 0.65

    # -- Data paths ----------------------------------------------------------
    REAL_DATA_DIR: str = _DEFAULT_REAL_DATA_DIR
    REAL_TRAIN_SUBFOLDERS: List[str] = field(default_factory=lambda: ["training"])
    BACKGROUND_DATA_DIR: str = ""   # set to load background class (Task 3)
    SYNTH_AUG_PASSES: int = 0
    REAL_AUG_PASSES: int = 60
    TRAIN_SPLIT_RATIO: float = 0.8

    # -- Real-time inference (Task 3) ----------------------------------------
    BACKGROUND_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.6
    VOTER_WINDOW_SIZE: int = 3

    # -- Output --------------------------------------------------------------
    MODEL_DIR: str = "./data/models/models_resnet/run"
    RANDOM_SEED: int = 42

    # -- W&B -----------------------------------------------------------------
    WANDB_PROJECT: str = "audio-classifier"
    WANDB_ENTITY: str = "bau-mat-ucl"

    def compute_n_frames(self) -> int:
        """
        Compute the number of time frames from Python DSP parameters only.
        This is INDEPENDENT of the MCU firmware configuration.
        """
        return int(self.DURATION_MS / 1000 * self.SAMPLE_RATE / self.HOP_LENGTH) + 1

    def to_json(self, path: str) -> None:
        """Save config as human-readable JSON (replaces model_config.pkl)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        # tuples are not JSON-serialisable by default — store as lists
        d["PITCH_SHIFT_SEMITONES"] = list(self.PITCH_SHIFT_SEMITONES)
        d["TIME_STRETCH_RANGE"] = list(self.TIME_STRETCH_RANGE)
        with open(path, "w", encoding="ascii") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BaseConfig":
        """Load config from JSON, with backward-compatible defaults."""
        with open(path, "r", encoding="ascii") as f:
            d = json.load(f)
        # Restore tuples
        if "PITCH_SHIFT_SEMITONES" in d:
            d["PITCH_SHIFT_SEMITONES"] = tuple(d["PITCH_SHIFT_SEMITONES"])
        if "TIME_STRETCH_RANGE" in d:
            d["TIME_STRETCH_RANGE"] = tuple(d["TIME_STRETCH_RANGE"])
        # Drop unknown keys for backward compat
        known = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    def wandb_config(self) -> dict:
        """Return a flat dict suitable for wandb.init(config=...)."""
        return dict(
            sample_rate=self.SAMPLE_RATE,
            n_mel=self.N_MEL,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            duration_ms=self.DURATION_MS,
            dropout=self.DROPOUT_RATE,
            l2_reg=self.L2_REG,
            label_smoothing=self.LABEL_SMOOTHING,
            mixup_alpha=self.MIXUP_ALPHA,
            activation=self.ACTIVATION,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            learning_rate=self.LEARNING_RATE,
            early_stop_patience=self.EARLY_STOPPING_PATIENCE,
            cosine_restart_epochs=self.COSINE_RESTART_EPOCHS,
            cosine_m_mul=self.COSINE_M_MUL,
            real_aug_passes=self.REAL_AUG_PASSES,
            synth_aug_passes=self.SYNTH_AUG_PASSES,
            background_threshold=self.BACKGROUND_THRESHOLD,
            confidence_threshold=self.CONFIDENCE_THRESHOLD,
        )
