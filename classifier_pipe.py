#GROUP_KEY = "dhhnIfhwZxTJCv7135lIm3zFtr96r3H3_xtKXRxU"import sys
import pickle
import numpy as np
import requests
import json
import librosa
import scipy.ndimage
from pathlib import Path
from datetime import datetime
import keras

"""
Usage : uv run auth --tcp-address tcp://127.0.0.1:10000 --no-authenticate | uv run python classifier_pipe.py
"""

HOSTNAME  = "http://localhost:5001"
GROUP_KEY = "HEwRwpUXlF3aTkpQusc4bMa30NCxhqWnHnjuPu05"
#GROUP_KEY = "dhhnIfhwZxTJCv7135lIm3zFtr96r3H3_xtKXRxU"
MODEL_PATH    = "classification/data/models/models_resnet/valacc9306_test85/best_model_valacc9306_test85.keras"
METADATA_PATH = "classification/data/models/models_resnet/valacc9306_test85/model_config.pkl"
PRINT_PREFIX  = "DF:HEX:"
GUESS_FILE    = "/tmp/latest_guess.json"

CONFIDENCE_THRESHOLD = 0.6
AUTO_SUBMIT = False

# Hardware ADC — fixes, lies au MCU
VAL_MAX_ADC       = 4096
FREQ_SAMPLING_MCU = 10200

# --- Chargement modele + metadata ---
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modele charge depuis {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"Erreur chargement modele : {e}", file=sys.stderr)
    sys.exit(1)

try:
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    print(f"Erreur chargement metadata : {e}", file=sys.stderr)
    sys.exit(1)

# Parametres DSP lus depuis le fichier de config sauvegarde a l'entrainement
classnames  = metadata["classnames"]
N_MELS      = metadata.get("n_mel",       64)
N_FFT       = metadata.get("n_fft",       512)
HOP_LENGTH  = metadata.get("hop_length",  128)
SAMPLE_RATE = metadata.get("sample_rate", 11025)
input_shape = metadata.get("input_shape", (N_MELS, 87, 1))

if len(input_shape) == 3:
    N_FRAMES = input_shape[1]
elif len(input_shape) == 2:
    N_FRAMES = input_shape[1]
elif len(input_shape) == 1:
    N_FRAMES = input_shape[0] // N_MELS
else:
    N_FRAMES = 87

print(f"Classes     : {classnames}", file=sys.stderr)
print(f"Input shape : {input_shape}  (N_MEL={N_MELS}, n_frames={N_FRAMES})", file=sys.stderr)
print(f"N_FFT={N_FFT}  hop={HOP_LENGTH}  SR={SAMPLE_RATE}Hz", file=sys.stderr)
if FREQ_SAMPLING_MCU != SAMPLE_RATE:
    print(f"Reechantillonnage {FREQ_SAMPLING_MCU} -> {SAMPLE_RATE} Hz", file=sys.stderr)


# --- Feature extraction identique a uart-reader.py ---
def raw_to_mel_features(data_uint16):
    """
    Meme pipeline que uart-reader.py :
    uint16 -> float -> centrage -> normalisation -> resample -> mel -> norm spectrale -> tensor
    """
    y = data_uint16.astype(np.float32)
    y -= np.mean(y)
    y /= (np.max(np.abs(y)) + 1e-9)

    if FREQ_SAMPLING_MCU != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=FREQ_SAMPLING_MCU, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmax=FREQ_SAMPLING_MCU // 2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / \
              (log_mel.std(axis=1, keepdims=True) + 1e-8)

    if log_mel.shape != (N_MELS, N_FRAMES):
        zoom    = [N_MELS / log_mel.shape[0], N_FRAMES / log_mel.shape[1]]
        log_mel = scipy.ndimage.zoom(log_mel, zoom, order=1)
    log_mel = log_mel[:N_MELS, :N_FRAMES]

    # Format selon le type de modele
    if len(input_shape) == 1:
        tensor = log_mel.flatten()[np.newaxis, :].astype(np.float32)
        expected = input_shape[0]
        if tensor.shape[1] > expected:   tensor = tensor[:, :expected]
        elif tensor.shape[1] < expected: tensor = np.pad(tensor, ((0,0),(0,expected-tensor.shape[1])))
    else:
        tensor = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

    return tensor


# --- Soumission / sauvegarde ---
def submit_guess(guess):
    url = f"{HOSTNAME}/lelec210x/leaderboard/submit/{GROUP_KEY}/{guess}"
    try:
        response = requests.post(url, timeout=1.0)
        print(f"Statut : {response.status_code} {response.reason}", file=sys.stderr)
        print(f"Reponse : {response.text}", file=sys.stderr)
        if response.ok:
            print(f"Succes : '{guess}' enregistre.", file=sys.stderr)
            return True
        return False
    except requests.exceptions.Timeout:
        print("Timeout serveur.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Erreur envoi : {e}", file=sys.stderr)
        return False


def save_guess_to_file(guess, probabilities):
    guess_data = {
        "value":         guess,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iso_timestamp": datetime.now().isoformat(),
        "probabilities": {
            c: float(round(p, 4))
            for c, p in zip(classnames, probabilities)
        },
        "confidence": float(round(float(np.max(probabilities)), 4))
    }
    try:
        with open(GUESS_FILE, "w") as f:
            json.dump(guess_data, f, indent=2)
        print(f"Guess '{guess}' sauvegarde dans {GUESS_FILE}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde : {e}", file=sys.stderr)
        return False


# --- Traitement d'une ligne UART ---
def process_line(line):
    line = line.strip()
    if not line.startswith(PRINT_PREFIX):
        return

    hex_payload = line[len(PRINT_PREFIX):]
    try:
        raw_bytes = bytes.fromhex(hex_payload)
        data = np.frombuffer(raw_bytes, dtype=np.dtype('<u2'))

        tensor        = raw_to_mel_features(data)
        probabilities = model.predict(tensor, verbose=0)[0]
        pred_idx      = np.argmax(probabilities)
        prediction    = classnames[pred_idx]
        confidence    = float(probabilities[pred_idx])

        print(f"\nSon detecte : {prediction} ({confidence:.2%})", file=sys.stderr)
        for c, p in zip(classnames, probabilities):
            bar = "#" * int(p * 20)
            print(f"  {c:<15}: {p:.4f}  {bar}", file=sys.stderr)

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Confiance insuffisante ({confidence:.2%} < {CONFIDENCE_THRESHOLD:.0%}) — ignore.", file=sys.stderr)
            return

        if AUTO_SUBMIT:
            submit_guess(prediction)
        else:
            save_guess_to_file(prediction, probabilities)

    except Exception as e:
        print(f"Erreur traitement : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


# --- Main ---
def main():
    print("=" * 60, file=sys.stderr)
    print("LELEC210X ResNet Classifier Pipeline", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Modele  : {MODEL_PATH}", file=sys.stderr)
    print(f"Classes : {classnames}", file=sys.stderr)
    print(f"Seuil   : {CONFIDENCE_THRESHOLD:.0%}", file=sys.stderr)
    print(f"Mode    : {'AUTO-SUBMIT' if AUTO_SUBMIT else 'MANUEL (fichier)'}", file=sys.stderr)
    if AUTO_SUBMIT:
        print(f"Serveur : {HOSTNAME}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("En attente de donnees depuis le pipe...", file=sys.stderr)

    for line in sys.stdin:
        process_line(line)


if __name__ == "__main__":
    main()