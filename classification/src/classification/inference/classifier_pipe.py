"""
Real-time classification pipeline: UART -> mel -> ResNet -> leaderboard.

Replaces the standalone uart-reader.py with a clean, modular version that:
  - Loads config.json (or model_config.pkl as fallback)
  - Uses MelExtractor for feature extraction
  - Uses RealTimeVoter for background filtering and majority voting
  - Submits only confident, non-background predictions

Usage:
    uv run python -m classification.inference.classifier_pipe \\
        --port /dev/cu.usbmodem... \\
        --model-dir classification/data/models/models_resnet/my_run \\
        --url http://leaderboard:5000 \\
        --key MY_KEY

Or pipe from auth:
    uv run auth | uv run python -m classification.inference.classifier_pipe \\
        --stdin --model-dir ...
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
import requests

import scipy.ndimage

from ..utils import payload_to_melvecs
from .predict import MelExtractor, load_config

PRINT_PREFIX_AUTH = "DF:HEX:"   # output of `uv run auth`
PRINT_PREFIX_UART = "SND:HEX:"  # direct UART raw-audio prefix

FREQ_SAMPLING_MCU = 10200
VAL_MAX_ADC = 4096
VDD = 3.3


# ---------------------------------------------------------------------------
# UART reading
# ---------------------------------------------------------------------------

def _read_uart(port: str):
    """Generator: yields raw uint16 audio buffers from UART (SND:HEX: prefix)."""
    import serial

    ser = serial.Serial(port=port, baudrate=115200)
    print(f"\nConnected to {port}. Waiting for data...\n")
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii", errors="ignore")
        line = line.strip()
        if line.startswith(PRINT_PREFIX_UART):
            raw_bytes = bytes.fromhex(line[len(PRINT_PREFIX_UART):])
            dt = np.dtype(np.uint16).newbyteorder("<")
            yield np.frombuffer(raw_bytes, dtype=dt)
        else:
            print(f"[MCU] {line}")


_PACKET_HEADER_LEN = 8
_PACKET_TAG_LEN = 16
_MELVEC_LENGTH = 20  # must match MCU config.h MELVEC_LENGTH


def _read_stdin(extractor: MelExtractor):
    """
    Generator: reads from stdin (pipe from `uv run auth`).

    auth outputs lines like: DF:HEX:<hex>
    The hex encodes N_MELVECS x MELVEC_LENGTH big-endian int16 mel vectors.
    Yields (N_MEL, n_frames, 1) float32 tensors ready for the model.
    """
    for line in sys.stdin:
        line = line.strip()
        if PRINT_PREFIX_AUTH not in line:
            continue
        idx = line.index(PRINT_PREFIX_AUTH)
        hex_payload = line[idx + len(PRINT_PREFIX_AUTH):]
        try:
            raw_bytes = bytes.fromhex(hex_payload.strip())
            n_bytes = len(raw_bytes)
            row_bytes = 2 * _MELVEC_LENGTH  # 40 bytes per mel vector row

            # If buffer includes the AES packet header+tag, strip them.
            # This handles direct MCU UART output (DF:HEX: includes full packet).
            if n_bytes % row_bytes != 0:
                stripped = raw_bytes[_PACKET_HEADER_LEN:-_PACKET_TAG_LEN]
                if len(stripped) % row_bytes == 0:
                    print(f"  [STDIN] {n_bytes}B → stripped header+tag → {len(stripped)}B",
                          file=sys.stderr)
                    raw_bytes = stripped
                    n_bytes = len(raw_bytes)
                else:
                    print(f"  [STDIN] unexpected payload size {n_bytes}B "
                          f"(not a multiple of {row_bytes})", file=sys.stderr)
                    continue

            print(f"  [STDIN] payload {n_bytes}B ({n_bytes // row_bytes} vectors)",
                  file=sys.stderr)
            melvecs = payload_to_melvecs(raw_bytes.hex())      # (melvec_len, n_melvecs) float
            melvecs = melvecs.astype(np.float32)
            melvecs = melvecs / (melvecs.max() + 1e-8)
            zoom = [extractor.cfg.N_MEL / melvecs.shape[0],
                    extractor.n_frames / melvecs.shape[1]]
            spec = scipy.ndimage.zoom(melvecs, zoom, order=1)
            yield spec[..., np.newaxis].astype(np.float32)     # (N_MEL, n_frames, 1)
        except Exception as e:
            print(f"  [STDIN] decode error: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------

def _save_wav(buf: np.ndarray, label: str, save_dir: str = "audio_files") -> str:
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    clean = "".join(c for c in label if c.isalnum() or c in "._- ")
    folder = os.path.join(save_dir, clean)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"acquisition_{timestamp}.wav")
    buf_f = buf.astype(np.float64)
    buf_f -= buf_f.mean()
    mx = np.max(np.abs(buf_f))
    if mx > 0:
        buf_f /= mx
    sf.write(path, buf_f, FREQ_SAMPLING_MCU)
    return path


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_classification_loop(
    source,
    model: keras.Model,
    extractor: MelExtractor,
    classnames: list,
    submit_url: str = None,
    key: str = None,
    save_audio: bool = True,
    show_plot: bool = True,
    pre_processed: bool = False,
) -> None:
    """
    Main loop: for each packet, classify and optionally submit.

    :param source:         iterable yielding either raw uint16 arrays (UART) or
                           (N_MEL, n_frames, 1) tensors (stdin/auth)
    :param pre_processed:  True when source yields ready tensors (stdin/auth path)
    :param submit_url:     leaderboard base URL (None = dry run)
    """
    if show_plot:
        import matplotlib.pyplot as plt
        n_mel = extractor.cfg.N_MEL
        n_frames = extractor.n_frames
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.tight_layout(pad=4)
        dummy = np.zeros((n_mel, n_frames))
        im_mel = ax2.imshow(dummy, origin="lower", aspect="auto",
                            cmap="viridis", interpolation="nearest")
        plt.colorbar(im_mel, ax=ax2, fraction=0.03, pad=0.04, label="dB (norm.)")

    counter = 0
    for raw_audio in source:
        counter += 1
        print(f"\n{'─'*52}")
        print(f"  Packet #{counter}")
        print(f"{'─'*52}")

        prediction = "unknown"
        confidence = 0.0
        mel_matrix = None
        probs = None

        try:
            tensor = raw_audio if pre_processed else extractor.from_mcu_packet(raw_audio, FREQ_SAMPLING_MCU)
            batch = tensor[np.newaxis, ...]
            probs = model.predict(batch, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            prediction = classnames[pred_idx]
            confidence = float(probs[pred_idx])

            print(f"  Raw prediction : {prediction.upper()}  ({confidence:.1%})")
            print("  Probabilities  :")
            for cls, p in zip(classnames, probs):
                bar = "X" * int(p * 20)
                print(f"    {cls:<14} {p:.2%}  {bar}")

            mel_matrix = tensor[:, :, 0]

        except Exception as e:
            print(f"  Classification error: {e}")

        # -- Submit every prediction -----------------------------------------
        if probs is not None and submit_url and key:
            try:
                resp = requests.post(
                    f"{submit_url}/lelec210x/leaderboard/submit/{key}/{prediction}"
                )
                print(f"  [HTTP]   SUBMIT {prediction.upper()} -> {resp.status_code}  {resp.text[:120]}")
            except Exception as e:
                print(f"  [HTTP]   Error: {e}")

        # -- Save audio ------------------------------------------------------
        if save_audio and not pre_processed:
            try:
                path = _save_wav(raw_audio, prediction)
                print(f"  WAV -> {path}")
            except Exception as e:
                print(f"  Save error: {e}")

        # -- Live plot -------------------------------------------------------
        if show_plot and mel_matrix is not None:
            try:
                import matplotlib.pyplot as plt
                ax1.clear()
                if not pre_processed:
                    times = np.linspace(0, len(raw_audio) / FREQ_SAMPLING_MCU,
                                        len(raw_audio))
                    voltage_mV = raw_audio * VDD / VAL_MAX_ADC * 1e3
                    ax1.plot(times, voltage_mV, lw=0.8)
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Voltage (mV)")
                    ax1.grid(alpha=0.3)
                else:
                    ax1.axis("off")
                ax1.set_title(f"Waveform  —  {prediction} ({confidence:.1%})",
                              fontsize=12)

                im_mel.set_data(mel_matrix)
                im_mel.set_clim(vmin=mel_matrix.min(), vmax=mel_matrix.max())
                ax2.set_title(
                    f"Mel  {prediction} ({confidence:.1%})  —  "
                    f"N_MEL={extractor.cfg.N_MEL}  n_frames={extractor.n_frames}",
                    fontsize=10,
                )
                plt.draw()
                plt.pause(0.001)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio classifier (MCU -> mel -> ResNet -> leaderboard)"
    )
    parser.add_argument("-p", "--port", default=None,
                        help="Serial port (e.g. /dev/cu.usbmodem...)")
    parser.add_argument("--stdin", action="store_true",
                        help="Read hex payloads from stdin (pipe from auth)")
    parser.add_argument("-m", "--model-dir", required=True,
                        help="Directory containing best_model.keras + config.json")
    parser.add_argument("--url", default=os.environ.get("LEADERBOARD_URL"),
                        help="Leaderboard base URL (env: LEADERBOARD_URL)")
    parser.add_argument("-k", "--key", default=os.environ.get("LEADERBOARD_KEY"),
                        help="Leaderboard private key (env: LEADERBOARD_KEY)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable saving .wav files")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live matplotlib plot")
    parser.add_argument("--bg-threshold", type=float, default=None,
                        help="Override background filter threshold (default: from config, usually 0.4)")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Override confidence threshold (default: from config, usually 0.6)")
    parser.add_argument("--voter-window", type=int, default=None,
                        help="Override voter window size (default: from config, usually 3)")
    args = parser.parse_args()

    if not args.port and not args.stdin:
        parser.error("Specify --port or --stdin")

    # -- Load model and config -----------------------------------------------
    model_dir = args.model_dir
    print(f"Loading model from {model_dir}...")
    cfg = load_config(model_dir)

    model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model.h5")
    model = keras.models.load_model(model_path)

    # Load classnames from config or metadata
    import pickle
    meta_path = os.path.join(model_dir, "model_config.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        classnames = meta["classnames"]
    else:
        raise FileNotFoundError(
            f"model_config.pkl not found in {model_dir}. "
            "Cannot determine classnames.")

    if args.bg_threshold is not None:
        cfg.BACKGROUND_THRESHOLD = args.bg_threshold
    if args.conf_threshold is not None:
        cfg.CONFIDENCE_THRESHOLD = args.conf_threshold
    if args.voter_window is not None:
        cfg.VOTER_WINDOW_SIZE = args.voter_window

    extractor = MelExtractor(cfg)

    print(f"  Classes     : {classnames}")
    print(f"  Input shape : {extractor.input_shape}")
    print(f"  BG threshold: {cfg.BACKGROUND_THRESHOLD}  "
          f"Conf threshold: {cfg.CONFIDENCE_THRESHOLD}  "
          f"Window: {cfg.VOTER_WINDOW_SIZE}")

    # -- Source --------------------------------------------------------------
    if args.stdin:
        source = _read_stdin(extractor)
        pre_processed = True
    else:
        source = _read_uart(args.port)
        pre_processed = False

    run_classification_loop(
        source=source,
        model=model,
        extractor=extractor,
        classnames=classnames,
        submit_url=args.url,
        key=args.key,
        save_audio=not args.no_audio,
        show_plot=not args.no_plot,
        pre_processed=pre_processed,
    )


if __name__ == "__main__":
    main()
