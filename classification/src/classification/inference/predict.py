"""
Inference utilities: feature extraction and real-time classification.

Classes:
    MelExtractor    -- .wav / raw audio -> (N_MEL, n_frames, 1) windows
    RealTimeVoter   -- sliding window voter with background filtering

Functions:
    predict_with_tta    -- re-exported from training.evaluate
    load_config         -- load BaseConfig from config.json (or .pkl fallback)
"""
from __future__ import annotations

import os
import pickle
from collections import Counter, deque
from typing import Optional

import librosa
import numpy as np

from ..configs.base_config import BaseConfig
from ..data.augmentation import AudioAugmentation
from ..utils.audio_student import AudioUtil


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class MelExtractor:
    """
    Convert a raw audio file into (N_MEL, n_frames, 1) mel-spectrogram windows.

    n_frames is computed entirely from Python DSP parameters — the MCU
    firmware (config.h) is never touched for this.
    """

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        self.n_frames = cfg.compute_n_frames()
        self.step_frames = max(1, self.n_frames // 2)

    @property
    def input_shape(self) -> tuple:
        return (self.cfg.N_MEL, self.n_frames, 1)

    def _compute_mel(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Normalised log-mel spectrogram, per-frequency zero-mean unit-var."""
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        mel = librosa.feature.melspectrogram(
            y=signal, sr=sr,
            n_mels=self.cfg.N_MEL,
            n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / \
                  (log_mel.std(axis=1, keepdims=True) + 1e-8)
        return log_mel

    def _sliding_windows(self, log_mel: np.ndarray) -> list:
        T = log_mel.shape[1]
        if T < self.n_frames:
            log_mel = np.pad(log_mel, ((0, 0), (0, self.n_frames - T)))
            T = self.n_frames
        windows = [
            log_mel[:, s:s + self.n_frames]
            for s in range(0, T - self.n_frames + 1, self.step_frames)
        ]
        return windows if windows else [log_mel[:, :self.n_frames]]

    def extract_from_audio(self, audio: tuple, augment: bool = False,
                           skip_noise: bool = False) -> list:
        """
        Extract mel windows from (signal, sr) tuple.

        :returns: list of (N_MEL, n_frames) arrays
        """
        sig, sr = audio
        if augment:
            sig, sr = AudioAugmentation.apply(
                (sig, sr), self.cfg, skip_noise=skip_noise)
        log_mel = self._compute_mel(sig, sr)
        if augment and self.cfg.ENABLE_SPEC_MASKING and np.random.random() > 0.5:
            log_mel = AudioUtil.spectro_aug_timefreq_masking(
                log_mel, max_mask_pct=0.15, n_freq_masks=3, n_time_masks=3)
        return self._sliding_windows(log_mel)

    def extract_from_file(self, path: str, augment: bool = False,
                          skip_noise: bool = False) -> list:
        """
        Load a .wav file and extract mel windows.

        :returns: list of (N_MEL, n_frames) arrays
        """
        aud = AudioUtil.open(path)
        aud = AudioUtil.resample(aud, self.cfg.SAMPLE_RATE)
        return self.extract_from_audio(aud, augment=augment, skip_noise=skip_noise)

    def from_mcu_packet(self, raw_uint16: np.ndarray,
                        mcu_sample_rate: int = 10200) -> np.ndarray:
        """
        Convert a raw MCU audio buffer (uint16) to a single (N_MEL, n_frames, 1)
        array ready for inference.

        :param raw_uint16: 1D array of uint16 ADC samples from the MCU
        :param mcu_sample_rate: MCU sampling rate (default: 10200 Hz)
        :returns: (N_MEL, n_frames, 1) float32 array
        """
        signal = raw_uint16.astype(np.float32)
        signal -= signal.mean()
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal /= max_abs

        # Resample to Python DSP sample rate if needed
        if mcu_sample_rate != self.cfg.SAMPLE_RATE:
            signal = librosa.resample(
                signal, orig_sr=mcu_sample_rate, target_sr=self.cfg.SAMPLE_RATE)

        windows = self.extract_from_audio((signal, self.cfg.SAMPLE_RATE),
                                          augment=False)
        # Return first window only (single-packet inference)
        return windows[0][..., np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# Config loader (with backward compat)
# ---------------------------------------------------------------------------

def load_config(model_dir: str) -> BaseConfig:
    """
    Load a BaseConfig from a saved model directory.

    Priority:
        1. config.json (new format)
        2. model_config.pkl (legacy)
    """
    json_path = os.path.join(model_dir, "config.json")
    if os.path.exists(json_path):
        return BaseConfig.from_json(json_path)

    pkl_path = os.path.join(model_dir, "model_config.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        cfg = BaseConfig()
        cfg.SAMPLE_RATE = meta.get("sample_rate", cfg.SAMPLE_RATE)
        cfg.N_MEL       = meta.get("n_mel",        cfg.N_MEL)
        cfg.N_FFT       = meta.get("n_fft",         cfg.N_FFT)
        cfg.HOP_LENGTH  = meta.get("hop_length",    cfg.HOP_LENGTH)
        return cfg

    raise FileNotFoundError(
        f"No config.json or model_config.pkl found in {model_dir}")


# ---------------------------------------------------------------------------
# Real-time voter (Task 3)
# ---------------------------------------------------------------------------

class RealTimeVoter:
    """
    Sliding-window majority voter for real-time classification.

    Implements the strategy from the contest sequence diagram:
        - Each MCU time slot (5s) sends ~3 packets.
        - A blank window (1-3s) of background noise separates time slots.
        - False submissions during blank windows cost 0.5 pts each.

    Decision logic per packet:
        1. If P(background) > bg_threshold          -> discard (blank window)
        2. If max(P(sound classes)) < conf_threshold -> discard (uncertain)
        3. Otherwise                                 -> add vote to window
        4. When window is full or reset is called    -> return majority class

    Usage:
        voter = RealTimeVoter(classnames=classnames, cfg=cfg)
        for packet in zmq_stream:
            probs = model.predict(preprocess(packet))
            result = voter.update(probs[0])
            if result is not None:
                submit(result)
    """

    def __init__(self, classnames: list, cfg: BaseConfig):
        self.classnames = classnames
        self.bg_threshold = cfg.BACKGROUND_THRESHOLD
        self.conf_threshold = cfg.CONFIDENCE_THRESHOLD
        self.window_size = cfg.VOTER_WINDOW_SIZE
        self.votes: deque = deque(maxlen=self.window_size)
        self._bg_idx: Optional[int] = (
            classnames.index("background")
            if "background" in classnames else None
        )

    def reset(self) -> None:
        """Clear the vote window (call between rounds if needed)."""
        self.votes.clear()

    def update(self, probs: np.ndarray) -> Optional[str]:
        """
        Process one packet's probability vector.

        :param probs: 1D array of shape (n_classes,)
        :returns: class name to submit, or None if no submission yet
        """
        # -- Background filter -----------------------------------------------
        if self._bg_idx is not None:
            if probs[self._bg_idx] > self.bg_threshold:
                return None  # blank window detected

            active = np.delete(probs, self._bg_idx)
            active_classes = [c for i, c in enumerate(self.classnames)
                              if i != self._bg_idx]
        else:
            active = probs
            active_classes = self.classnames

        # -- Confidence filter -----------------------------------------------
        if active.max() < self.conf_threshold:
            return None  # too uncertain

        # -- Record vote -----------------------------------------------------
        best_cls = active_classes[int(np.argmax(active))]
        self.votes.append(best_cls)

        # -- Majority decision -----------------------------------------------
        if len(self.votes) == self.window_size:
            majority = Counter(self.votes).most_common(1)[0][0]
            self.votes.clear()
            return majority

        return None
