import argparse
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import serial
import soundfile as sf
import librosa
import librosa.effects
import scipy.ndimage
from pathlib import Path
from serial.tools import list_ports
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras

PRINT_PREFIX = "SND:HEX:"
project_root  = Path(__file__).resolve().parents[2]
MODEL_PATH    = project_root / "classification" / "data" / "models" / "models_resnet" / "valacc9306_test85" / "best_model_valacc9306_test85.keras"
METADATA_PATH = project_root / "classification" / "data" / "models" / "models_resnet" / "valacc9306_test85" / "model_config.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_PATH.with_suffix(".h5")

print("Chargement du modele et de la configuration...", end=" ", flush=True)
try:
    model = keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    print(f"\n  Erreur : {e}")
    sys.exit(1)

classnames  = metadata["classnames"]
N_MELS      = metadata.get("n_mel",       20)
N_FFT       = metadata.get("n_fft",       1024)
HOP_LENGTH  = metadata.get("hop_length",  256)
SAMPLE_RATE = metadata.get("sample_rate", 22050)
input_shape = metadata.get("input_shape", (N_MELS, 20, 1))

if len(input_shape) == 3:
    N_FRAMES = input_shape[1]
elif len(input_shape) == 2:
    N_FRAMES = input_shape[1]
elif len(input_shape) == 1:
    total = input_shape[0]
    N_FRAMES = total // N_MELS if N_MELS > 0 else 20
else:
    N_FRAMES = 20

VAL_MAX_ADC       = 4096
VDD               = 3.3
FREQ_SAMPLING_MCU = 10200

print("OK")
print(f"  Classes     : {classnames}")
print(f"  Input shape : {input_shape}  (N_MEL={N_MELS}, n_frames={N_FRAMES})")
print(f"  N_FFT       : {N_FFT}  |  Hop : {HOP_LENGTH}  |  SR : {SAMPLE_RATE} Hz")
if FREQ_SAMPLING_MCU != SAMPLE_RATE:
    print(f"  Reechantillonnage {FREQ_SAMPLING_MCU} -> {SAMPLE_RATE} Hz")


# ── Augmentations (standalone, sans Config) ───────────────────
def aug_time_shift(sig, sr, shift_max=10.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(sig))
    return np.roll(sig, shift)

def aug_pitch_shift(sig, sr, n_steps_range=(-3, 3)):
    n = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(sig, sr=sr, n_steps=n)

def aug_time_stretch(sig, sr, rate_range=(0.25, 1.85)):
    rate = np.random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(sig, rate=rate)
    if len(stretched) > len(sig):
        return stretched[:len(sig)]
    elif len(stretched) < len(sig):
        return np.pad(stretched, (0, len(sig) - len(stretched)))
    return stretched


# ── Feature extraction commune ────────────────────────────────
def compute_mel(sig, sr):
    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    mel = librosa.feature.melspectrogram(
        y=sig, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmax=FREQ_SAMPLING_MCU // 2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / \
              (log_mel.std(axis=1, keepdims=True) + 1e-8)
    if log_mel.shape[1] < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, N_FRAMES - log_mel.shape[1])))
    return log_mel[:, :N_FRAMES]


def raw_to_mel_features(audio_buffer):
    y = np.array(audio_buffer, dtype=np.float32)
    y -= np.mean(y)
    y /= (np.max(np.abs(y)) + 1e-9)
    if FREQ_SAMPLING_MCU != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=FREQ_SAMPLING_MCU, target_sr=SAMPLE_RATE)
    log_mel = compute_mel(y, SAMPLE_RATE)
    if log_mel.shape != (N_MELS, N_FRAMES):
        zoom    = [N_MELS / log_mel.shape[0], N_FRAMES / log_mel.shape[1]]
        log_mel = scipy.ndimage.zoom(log_mel, zoom, order=1)
    log_mel = log_mel[:N_MELS, :N_FRAMES]
    if len(input_shape) == 1:
        tensor = log_mel.flatten()[np.newaxis, :].astype(np.float32)
        expected = input_shape[0]
        if tensor.shape[1] > expected:   tensor = tensor[:, :expected]
        elif tensor.shape[1] < expected: tensor = np.pad(tensor, ((0,0),(0, expected-tensor.shape[1])))
    else:
        tensor = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
    return tensor, log_mel


# ── Visualisation augmentations depuis buffer MCU ─────────────
def plot_augmentations_from_buffer(raw_audio, prediction, confidence, save_dir="audio_files"):
    """
    Adapte plot_augmentations pour travailler directement depuis
    le buffer uint16 recu par UART, sans passer par un fichier.
    """
    y = np.array(raw_audio, dtype=np.float32)
    y -= np.mean(y)
    y /= (np.max(np.abs(y)) + 1e-9)
    if FREQ_SAMPLING_MCU != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=FREQ_SAMPLING_MCU, target_sr=SAMPLE_RATE)
    sr = SAMPLE_RATE

    mel_orig = compute_mel(y, sr)
    mel_ts   = compute_mel(aug_time_shift(y.copy(),   sr), sr)
    mel_ps   = compute_mel(aug_pitch_shift(y.copy(),  sr), sr)
    mel_tt   = compute_mel(aug_time_stretch(y.copy(), sr), sr)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)

    aug_data   = [mel_ts,  mel_ps,       mel_tt]
    aug_titles = ["Time shift", "Pitch shift", "Time stretch"]

    # Plot "Original" mel spectrogram, spanning all rows and first two columns (wider)
    ax_orig = fig.add_subplot(gs[:, :3])
    im_orig = ax_orig.imshow(mel_orig, origin="lower", aspect="auto",
                            cmap="viridis", interpolation="nearest")
    ax_orig.set_title("Original", fontsize=13, fontweight='bold')
    ax_orig.set_xlabel("Frames")
    ax_orig.set_ylabel("Bandes mel")

    # Add colorbar for amplitude scale
    cbar = fig.colorbar(im_orig, ax=ax_orig, fraction=0.03, pad=0.04, label="Amplitude (dB, norm.)")

    # Plot augmentations in columns 2, 3, 4 (one per row)
    for row, (mel, title, cmap) in enumerate(zip(aug_data, aug_titles, ["viridis", "viridis", "viridis"])):
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(mel, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Frames")
        ax.set_ylabel("Bandes mel")

    fig.suptitle(
        f"Augmentations — {prediction} ({confidence:.1%})  |  "
        f"SR={SAMPLE_RATE}Hz  N_FFT={N_FFT}  hop={HOP_LENGTH}",
        fontsize=12
    )

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(save_dir, prediction,
                             f"augmentations_{timestamp}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Augmentations -> {save_path}")
    plt.close(fig)


# ── Parsing UART ──────────────────────────────────────────────
def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    print(f"[MCU] {line}")
    return None

def reader(port):
    ser = serial.Serial(port=port, baudrate=115200)
    print(f"\nConnecte a {port}. En attente de donnees...\n")
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii", errors="ignore")
        buf = parse_buffer(line.strip())
        if buf is not None:
            dt = np.dtype(np.uint16).newbyteorder("<")
            yield np.frombuffer(buf, dtype=dt)

def generate_audio_file(buf, label):
    timestamp   = datetime.now().strftime("%m_%d_%H_%M_%S")
    clean_label = "".join(x for x in str(label) if x.isalnum() or x in "._- ")
    folder_path = os.path.join("audio_files", clean_label)
    os.makedirs(folder_path, exist_ok=True)
    full_path   = os.path.join(folder_path, f"acquisition_{timestamp}.wav")
    buf_float   = np.asarray(buf, dtype=np.float64)
    buf_float  -= np.mean(buf_float)
    mx = np.max(np.abs(buf_float))
    if mx > 0:
        buf_float /= mx
    sf.write(full_path, buf_float, FREQ_SAMPLING_MCU)
    return full_path


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port")
    argParser.add_argument("--aug", action="store_true",
                           help="Sauvegarder le plot d'augmentations a chaque acquisition")
    args = argParser.parse_args()

    if args.port is None:
        print("Ports disponibles :")
        for p in list(list_ports.comports()):
            print(f"  - {p.device}")
        print("Usage : uv run uart-reader.py -p /dev/cu.usbmodem... [--aug]")
        sys.exit(0)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.tight_layout(pad=4)
    dummy  = np.zeros((N_MELS, N_FRAMES))
    im_mel = ax2.imshow(dummy, origin="lower", aspect="auto",
                        cmap="viridis", interpolation="nearest")
    cbar   = plt.colorbar(im_mel, ax=ax2, fraction=0.03, pad=0.04, label="dB (norm.)")

    msg_counter = 0
    for raw_audio in reader(port=args.port):
        msg_counter += 1
        print(f"\n{'─'*50}")
        print(f"  Acquisition #{msg_counter}")
        print(f"{'─'*50}")

        prediction = "unknown"
        confidence = 0.0
        mel_matrix = np.zeros((N_MELS, N_FRAMES))
        try:
            tensor, mel_matrix = raw_to_mel_features(raw_audio)
            probs      = model.predict(tensor, verbose=0)[0]
            pred_idx   = np.argmax(probs)
            prediction = classnames[pred_idx]
            confidence = probs[pred_idx]
            print(f"  -> {prediction.upper()}  ({confidence:.1%})")
            print("  Probabilites :")
            for cls, p in zip(classnames, probs):
                print(f"    {cls:<12} {p:.2%}  {'X' * int(p * 20)}")
        except Exception as e:
            print(f"  Erreur classification : {e}")

        path = generate_audio_file(raw_audio, prediction)
        print(f"  WAV -> {path}")

        # Plot augmentations si flag --aug active
        if args.aug:
            plot_augmentations_from_buffer(raw_audio, prediction, confidence)

        ax1.clear()
        times      = np.linspace(0, len(raw_audio) / FREQ_SAMPLING_MCU, len(raw_audio))
        voltage_mV = raw_audio * VDD / VAL_MAX_ADC * 1e3
        ax1.plot(times, voltage_mV, lw=0.8)
        ax1.set_title(f"Waveform  --  {prediction} ({confidence:.1%})", fontsize=12)
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Tension (mV)")
        ax1.grid(alpha=0.3)

        ax2.clear()
        im_mel.set_data(mel_matrix)
        im_mel.set_clim(vmin=mel_matrix.min(), vmax=mel_matrix.max())
        cbar.update_normal(im_mel)
        ax2.add_image(im_mel)
        ax2.set_title(
            f"Mel-spectrogramme {prediction} ({confidence:.1%})  --  "
            f"{N_MELS} bandes x {N_FRAMES} frames  "
            f"(N_FFT={N_FFT}, hop={HOP_LENGTH}, SR={SAMPLE_RATE} Hz, "
            f"fmax={FREQ_SAMPLING_MCU//2} Hz)",
            fontsize=10
        )
        ax2.set_xlabel("Frames temporelles")
        ax2.set_ylabel("Bandes mel")
        ax2.set_xlim(0, N_FRAMES - 1)
        ax2.set_ylim(0, N_MELS - 1)
        plt.draw()
        plt.pause(0.001)