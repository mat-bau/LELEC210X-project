"""
compare_norm.py — Compare normalisation strategies for mel spectrograms.

Columns:
  1. OLD: per-band z-score       (noisy — silent bands amplified)
  2. NEW: fixed dB / 80          (preserves inter-band energy contrast)
  3. audio_student log10         (reference pipeline from notebook)
  4. L2 norm (Frobenius)         (amplitude-invariant, KNN-style from h5a2)
  5. MCU linear magnitude        (what the MCU actually sends via radio)
  6. Difference (fixed dB − per-band z-score)

MCU pipeline (spectrogram.c):
  ADC (12-bit) → shift<<3, DC remove → Hamming window → arm_rfft_q15
  → normalize by vmax (overflow trick, undone after) → |magnitude|
  → mel filterbank (hz2mel_mat) → Q1.15 int16 output
  KEY: NO log, NO per-frame normalisation of mel output.
  "vmax trick" only prevents Q1.15 overflow during magnitude computation.

Usage (from project root):
    cd classification
    uv run python compare_norm.py
    # → saves compare_norm.png in the current directory
"""
import os
import sys
import glob

import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from classification.utils.audio_student import AudioUtil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 10200
N_MEL       = 32
N_FFT       = 512
HOP_LENGTH  = 512
DB_RANGE    = 80.0

AUDIO_ROOT = os.path.join(
    os.path.dirname(__file__), "..",
    "mcu", "hands_on_audio_acquisition", "audio_files",
)
CLASSES = ["chainsaw", "fire", "fireworks", "gunshot"]


# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------
def _load_raw(path: str):
    """Load and peak-normalise the waveform."""
    sig, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    return sig, sr


def _power_to_db(sig, sr) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=sig, sr=sr, n_mels=N_MEL, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)  # range [-80, 0]


def norm_per_band(log_mel: np.ndarray) -> np.ndarray:
    """OLD: per-band z-score along time axis (independent per frequency band).
    Amplifies noise in silent bands → uniform noisy appearance.
    """
    return (log_mel - log_mel.mean(axis=1, keepdims=True)) / \
           (log_mel.std(axis=1, keepdims=True) + 1e-8)


def norm_fixed_db(log_mel: np.ndarray) -> np.ndarray:
    """NEW: fixed scaling [-80, 0] dB → [-1, 0].
    power_to_db(ref=np.max) guarantees peak=0 dB → dividing by 80 is deterministic.
    Preserves inter-band contrast; no per-sample statistics needed.
    """
    return log_mel / DB_RANGE


def norm_audio_student(path: str) -> np.ndarray:
    """audio_student.py pipeline: log10(magnitude_mel), no normalisation."""
    aud = AudioUtil.open(path)
    aud = AudioUtil.resample(aud, SAMPLE_RATE)
    return AudioUtil.melspectrogram(aud, Nmel=N_MEL, Nft=N_FFT, fs2=SAMPLE_RATE)


def norm_l2(log_mel: np.ndarray) -> np.ndarray:
    """L2 norm (Frobenius): divide the whole spectrogram by its L2 norm.
    Equivalent to the KNN approach in h5a2_audio.ipynb on the flattened feature vector:
        X_normalised = X / np.linalg.norm(X, axis=1, keepdims=True)
    Here applied to the 2D spectrogram flatten → unit-norm feature vector.
    Amplitude-invariant: two identical sound patterns at different volumes → same result.
    Note: power_to_db(ref=np.max) already partially handles amplitude (peak always 0 dB),
    so L2 norm here adds full invariance but changes per-sample scale.
    """
    return log_mel / (np.linalg.norm(log_mel) + 1e-8)


def norm_mcu_linear(sig: np.ndarray, sr: int) -> np.ndarray:
    """Simulate the MCU DSP pipeline (spectrogram.c).

    MCU pipeline per MELVEC:
      1. Hamming window on SAMPLES_PER_MELVEC=512 samples (non-overlapping)
      2. arm_rfft_q15: FFT on real signal
      3. Normalize by vmax, compute |magnitude|, denormalize  ← OVERFLOW TRICK ONLY,
         the amplitude IS preserved in the output
      4. mel filterbank (hz2mel_mat @ magnitude)
      Output: Q1.15 int16, linear mel magnitude in [0, 32767]

    Python equivalent:
      - librosa mel spectrogram with Hamming window, power=1 (magnitude, not power)
      - hop_length = n_fft = 512 (non-overlapping frames, same as MCU)
      - No log, no per-band normalisation

    Displayed here normalised to [0, 1] by dividing by global max for visual
    comparison — the MCU sends the raw Q1.15 int16 values (0..32767).
    """
    mel_mag = librosa.feature.melspectrogram(
        y=sig, sr=sr,
        n_mels=N_MEL,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window='hamming',
        power=1,          # linear magnitude spectrum (not power)
    )
    # Normalise to [0, 1] for display only (real MCU divides by 32768 for Q1.15 float)
    return mel_mag / (mel_mag.max() + 1e-8)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------
def _one_file(cls: str) -> str:
    pattern = os.path.join(AUDIO_ROOT, cls, "training", "*.wav")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(AUDIO_ROOT, cls, "*.wav")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .wav found for class {cls} in {AUDIO_ROOT}")
    file = files[74]
    print(file)
    return file


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def main():
    n_cls  = len(CLASSES)
    n_cols = 6   # per-band | fixed-dB | audio_student | L2 | MCU linear | difference
    col_titles = [
        "OLD: per-band z-score\n(noisy — silent bands boosted)",
        "NEW: fixed dB / 80\n(preserves freq contrast)",
        "audio_student log10\n(notebook reference)",
        "L2 norm (Frobenius)\n(amplitude-invariant, KNN-style)",
        "MCU: linear magnitude\n(sqrt power, Hamming, hop=512\nno log — what radio sends)",
        "Difference\n(fixed dB − per-band z-score)",
    ]

    fig, axes = plt.subplots(n_cls, n_cols,
                             figsize=(n_cols * 3.5, n_cls * 3 + 1))
    fig.suptitle("Mel Spectrogram Normalisation Comparison", fontsize=14,
                 fontweight="bold", y=1.01)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=6)

    for row, cls in enumerate(CLASSES):
        path = _one_file(cls)
        sig, sr = _load_raw(path)
        log_mel = _power_to_db(sig, sr)

        # Truncate to first n_frames frames (as the model would see)
        n_frames = int(1557 / 1000 * SAMPLE_RATE / HOP_LENGTH) + 1  # = 32
        log_mel = log_mel[:, :n_frames]

        s_perband = norm_per_band(log_mel)
        s_fixeddb = norm_fixed_db(log_mel)
        s_audstu  = norm_audio_student(path)[:, :n_frames]
        s_l2      = norm_l2(log_mel)
        s_mcu     = norm_mcu_linear(sig, sr)[:, :n_frames]
        s_diff    = s_fixeddb - s_perband

        specs = [s_perband, s_fixeddb, s_audstu, s_l2,   s_mcu,   s_diff]
        cmaps = ["magma",   "magma",   "magma",  "magma", "magma", "RdBu_r"]

        for col, (spec, cmap) in enumerate(zip(specs, cmaps)):
            ax = axes[row, col]
            is_diff = (col == n_cols - 1)  # last column = difference
            if is_diff:
                vabs = max(abs(s_diff.min()), abs(s_diff.max()))
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = spec.min(), spec.max()
            im = ax.imshow(spec, aspect="auto", origin="lower",
                           cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            if col == 0:
                ax.set_ylabel(f"{cls}\n\nMel bands", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time frames")
            ax.set_xticks([])
            ax.set_yticks([])

        # Print stats
        print(f"\n{cls} ({os.path.basename(path)}):")
        print(f"  per-band  : min={s_perband.min():.3f}  max={s_perband.max():.3f}"
              f"  std={s_perband.std():.4f}")
        print(f"  fixed-db  : min={s_fixeddb.min():.3f}  max={s_fixeddb.max():.3f}"
              f"  std={s_fixeddb.std():.4f}")
        print(f"  audio_stu : min={s_audstu.min():.3f}  max={s_audstu.max():.3f}"
              f"  std={s_audstu.std():.4f}")
        print(f"  l2-norm   : min={s_l2.min():.3f}  max={s_l2.max():.3f}"
              f"  std={s_l2.std():.4f}  |norm|={np.linalg.norm(s_l2):.4f}")
        print(f"  MCU-linear: min={s_mcu.min():.3f}  max={s_mcu.max():.3f}"
              f"  std={s_mcu.std():.4f}  (linear magnitude, Hamming, hop=512)")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "compare_norm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
