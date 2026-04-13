"""
Audio augmentation pipeline and MixUp generator.

AudioAugmentation — static methods for waveform augmentation.
MixupGenerator    — Keras Sequence that applies MixUp on-the-fly.
"""
from __future__ import annotations

import numpy as np
import librosa
import librosa.effects

import keras
from keras.utils import to_categorical

from ..configs.base_config import BaseConfig
from ..utils.audio_student import AudioUtil


class AudioAugmentation:
    """Static augmentation methods for raw audio signals."""

    @staticmethod
    def time_shift(signal: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """Random circular shift in time."""
        shift = int(np.random.uniform(-shift_max, shift_max) * len(signal))
        out = np.zeros_like(signal)
        if shift > 0:
            out[shift:] = signal[:-shift]
        elif shift < 0:
            out[:shift] = signal[-shift:]
        else:
            out = signal.copy()
        return out

    @staticmethod
    def pitch_shift(signal: np.ndarray, sr: int, semitone_range: tuple) -> np.ndarray:
        """Pitch shift via librosa."""
        n_steps = np.random.uniform(*semitone_range)
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

    @staticmethod
    def time_stretch(signal: np.ndarray, rate_range: tuple) -> np.ndarray:
        """Time stretch with padding / truncation to preserve length."""
        rate = np.random.uniform(*rate_range)
        stretched = librosa.effects.time_stretch(signal, rate=rate)
        if len(stretched) > len(signal):
            return stretched[:len(signal)]
        return np.pad(stretched, (0, len(signal) - len(stretched)))

    @staticmethod
    def apply(audio: tuple, cfg: BaseConfig, skip_noise: bool = False) -> tuple:
        """Apply all enabled augmentations with 50% probability each."""
        sig, sr = audio
        if cfg.ENABLE_TIME_SHIFT and np.random.random() > 0.5:
            sig = AudioAugmentation.time_shift(sig)
        if cfg.ENABLE_PITCH_SHIFT and np.random.random() > 0.5:
            sig = AudioAugmentation.pitch_shift(sig, sr, cfg.PITCH_SHIFT_SEMITONES)
        if cfg.ENABLE_TIME_STRETCH and np.random.random() > 0.5:
            sig = AudioAugmentation.time_stretch(sig, cfg.TIME_STRETCH_RANGE)
        if not skip_noise and cfg.ENABLE_NOISE and np.random.random() > 0.5:
            sig, sr = AudioUtil.add_noise((sig, sr), sigma=cfg.NOISE_SIGMA)
        return sig, sr


def _mixup(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple:
    """Beta-weighted linear interpolation of samples and one-hot labels."""
    if alpha == 0:
        return X, y
    lam = np.random.beta(alpha, alpha)
    perm = np.random.permutation(len(X))
    return lam * X + (1 - lam) * X[perm], lam * y + (1 - lam) * y[perm]


class MixupGenerator(keras.utils.Sequence):
    """Batch generator that applies MixUp on-the-fly during training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int,
                 n_classes: int, alpha: float = 0.3, shuffle: bool = True):
        self.X = X
        self.y_cat = to_categorical(y, n_classes)
        self.bs = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.idx = np.arange(len(X))

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, i: int) -> tuple:
        batch = self.idx[i * self.bs:(i + 1) * self.bs]
        Xb, yb = _mixup(self.X[batch], self.y_cat[batch], self.alpha)
        return Xb, yb

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.idx)
