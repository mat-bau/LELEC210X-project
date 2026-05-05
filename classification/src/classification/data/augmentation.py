"""
Audio augmentation pipeline and MixUp generator.

AudioAugmentation — static methods for waveform augmentation.
MixupGenerator    — Keras Sequence that applies MixUp on-the-fly.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import librosa
import librosa.effects

import keras
from keras.utils import to_categorical

from ..configs.base_config import BaseConfig
from ..utils.audio_student import AudioUtil

# Module-level cache: bg_dir -> list of .wav paths (avoids repeated glob)
_BG_FILE_CACHE: dict = {}


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
    def add_real_background(signal: np.ndarray, sr: int,
                            cfg: BaseConfig) -> np.ndarray:
        """
        Mix a random chunk of a real background recording into signal.

        Scans cfg.BACKGROUND_DATA_DIR recursively for .wav files (result is
        cached per directory). Picks one file at random, extracts a random
        contiguous chunk of the same length as signal, and adds it at an
        amplitude drawn uniformly from cfg.BG_OVERLAY_AMP.
        """
        bg_dir = getattr(cfg, "BACKGROUND_DATA_DIR", "")
        if not bg_dir or not os.path.isdir(bg_dir):
            return signal

        if bg_dir not in _BG_FILE_CACHE:
            _BG_FILE_CACHE[bg_dir] = glob.glob(
                os.path.join(bg_dir, "**", "*.wav"), recursive=True
            )

        bg_files = _BG_FILE_CACHE[bg_dir]
        if not bg_files:
            return signal

        bg_path = bg_files[np.random.randint(len(bg_files))]
        try:
            bg_sig, _ = librosa.load(bg_path, sr=sr, mono=True)
        except Exception:
            return signal

        n = len(signal)
        if len(bg_sig) < n:
            # Loop until long enough
            repeats = int(np.ceil(n / len(bg_sig)))
            bg_sig = np.tile(bg_sig, repeats)

        start = np.random.randint(0, len(bg_sig) - n + 1)
        bg_chunk = bg_sig[start:start + n]

        amp_min, amp_max = getattr(cfg, "BG_OVERLAY_AMP", (0.05, 0.35))
        amp = float(np.random.uniform(amp_min, amp_max))
        return signal + amp * bg_chunk

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
        if (getattr(cfg, "ENABLE_BG_OVERLAY", False)
                and getattr(cfg, "BACKGROUND_DATA_DIR", "")
                and np.random.random() > 0.5):
            sig = AudioAugmentation.add_real_background(sig, sr, cfg)
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
