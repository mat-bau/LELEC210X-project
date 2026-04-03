"""
classifier_pipe.py
LELEC210X — ResNet Classifier Pipeline (GNU Radio pipe)

Usage:
    uv run auth --tcp-address tcp://127.0.0.1:10000 --no-authenticate \
    | uv run python classifier_pipe.py

Principe :
    Le MCU calcule et envoie un mel-spectrogramme PRÉ-CALCULÉ de taille
    (n_melvecs, melvec_length) — ex: (64, 20) ou (32, 20).
    Ce script le reçoit, le redimensionne automatiquement vers la taille
    attendue par le modèle (N_MEL, N_FRAMES), et fait la prédiction.

    La taille envoyée par le MCU peut être DIFFÉRENTE de celle du modèle :
    le zoom scipy.ndimage s'en charge automatiquement.
    Exemple typique :
        MCU envoie  : (64, 20)  → transposé → (20, 64)
        Modèle veut : (64, 87)
        Zoom appliqué : x3.2 sur axe fréquence, x1.36 sur axe temps
"""

import sys
import pickle
import numpy as np
import requests
import json
import scipy.ndimage
from pathlib import Path
from datetime import datetime
import keras

# ── Configuration ─────────────────────────────────────────────
GROUP_KEY  = "dhhnIfhwZxTJCv7135lIm3zFtr96r3H3_xtKXRxU"
HOSTNAME   = "http://lelec210x.sipr.ucl.ac.be/lelec210x"

MODEL_PATH    = "classification/data/models/models_resnet/valacc8423_test8938_nodataaug/best_model.keras"
METADATA_PATH = "classification/data/models/models_resnet/valacc8423_test8938_nodataaug/model_config.pkl"

PRINT_PREFIX = "DF:HEX:"
GUESS_FILE   = "/tmp/latest_guess.json"

CONFIDENCE_THRESHOLD = 0.51
AUTO_SUBMIT          = False

# ── Paramètres MCU — CE QUE LE MCU ENVOIE RÉELLEMENT ─────────
# À modifier ici si tu changes la config GNU Radio/MCU,
# indépendamment de ce sur quoi le modèle a été entraîné.
MCU_N_MELVECS     = 64   # nombre de frames temporelles envoyées par le MCU
MCU_MELVEC_LENGTH = 20   # nombre de bandes mel envoyées par le MCU

# ── Chargement modèle ─────────────────────────────────────────
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modèle chargé depuis {MODEL_PATH}", file=sys.stderr)
except Exception as e:
    print(f"Erreur chargement modèle : {e}", file=sys.stderr)
    sys.exit(1)

try:
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
except Exception as e:
    print(f"Erreur chargement metadata : {e}", file=sys.stderr)
    sys.exit(1)

# ── Paramètres DSP lus depuis le pkl (ce sur quoi le modèle a été entraîné) ──
classnames  = metadata["classnames"]
MODEL_N_MEL = metadata.get("n_mel",       64)
input_shape = metadata.get("input_shape", (MODEL_N_MEL, 87, 1))

# Dimensions attendues par le modèle
if len(input_shape) == 3:
    MODEL_N_FRAMES = input_shape[1]   # (N_MEL, N_FRAMES, 1)
elif len(input_shape) == 2:
    MODEL_N_FRAMES = input_shape[1]   # (N_MEL, N_FRAMES)
elif len(input_shape) == 1:
    MODEL_N_FRAMES = input_shape[0] // MODEL_N_MEL
else:
    MODEL_N_FRAMES = 87

# Taille brute reçue du MCU après transposition (freq, time)
MCU_SHAPE   = (MCU_MELVEC_LENGTH, MCU_N_MELVECS)   # ex: (20, 64)
MODEL_SHAPE = (MODEL_N_MEL, MODEL_N_FRAMES)          # ex: (64, 87)

# Facteurs de zoom calculés automatiquement
ZOOM_FREQ = MODEL_N_MEL    / MCU_MELVEC_LENGTH   # ex: 64/20  = 3.2
ZOOM_TIME = MODEL_N_FRAMES / MCU_N_MELVECS        # ex: 87/64  = 1.36

print("=" * 60, file=sys.stderr)
print("LELEC210X ResNet Classifier Pipeline", file=sys.stderr)
print("=" * 60, file=sys.stderr)
print(f"Modèle attend  : {MODEL_SHAPE}  (N_MEL={MODEL_N_MEL}, N_FRAMES={MODEL_N_FRAMES})", file=sys.stderr)
print(f"MCU envoie     : {MCU_SHAPE}   (melvec_length={MCU_MELVEC_LENGTH}, n_melvecs={MCU_N_MELVECS})", file=sys.stderr)

if MCU_SHAPE == MODEL_SHAPE:
    print("Zoom           : aucun (MCU et modèle ont la même taille)", file=sys.stderr)
else:
    print(f"Zoom auto      : ×{ZOOM_FREQ:.2f} fréquence, ×{ZOOM_TIME:.2f} temps", file=sys.stderr)
    print(f"               : {MCU_SHAPE} → {MODEL_SHAPE}", file=sys.stderr)

print(f"Classes        : {classnames}", file=sys.stderr)
print(f"Seuil          : {CONFIDENCE_THRESHOLD:.0%}", file=sys.stderr)
print(f"Mode           : {'AUTO-SUBMIT' if AUTO_SUBMIT else 'MANUEL (fichier)'}", file=sys.stderr)
print("=" * 60, file=sys.stderr)


# ── Feature extraction depuis mel MCU ─────────────────────────
def mcu_mel_to_tensor(data_uint16):
    """
    Transforme les données brutes uint16 reçues du MCU en tenseur
    compatible avec le ResNet.

    Pipeline :
    1. Reshape (N,) → (n_melvecs, melvec_length) — format MCU
    2. Transposition → (melvec_length, n_melvecs) = (freq, time)
    3. Conversion float + normalisation spectrale (mean/std par bande)
    4. Zoom vers (MODEL_N_MEL, MODEL_N_FRAMES) si nécessaire
    5. Ajout dimensions batch + canal → (1, N_MEL, N_FRAMES, 1)

    Paramètres
    ----------
    data_uint16 : np.ndarray, shape (MCU_N_MELVECS * MCU_MELVEC_LENGTH,)
        Données brutes reçues via UART, dtype uint16.

    Retourne
    --------
    tensor : np.ndarray, shape (1, MODEL_N_MEL, MODEL_N_FRAMES, 1)
    mel_2d : np.ndarray, shape (MODEL_N_MEL, MODEL_N_FRAMES)
        Le spectrogramme après zoom, pour affichage/debug.
    """
    expected_len = MCU_N_MELVECS * MCU_MELVEC_LENGTH

    # 1. Reshape : (N,) → (n_melvecs, melvec_length)
    mel = data_uint16[:expected_len].astype(np.float32)
    mel = mel.reshape(MCU_N_MELVECS, MCU_MELVEC_LENGTH)

    # 2. Transposition : (n_melvecs, melvec_length) → (freq, time)
    #    Le MCU envoie (temps, fréquence), le modèle attend (fréquence, temps)
    mel = mel.T   # → (MCU_MELVEC_LENGTH, MCU_N_MELVECS) = (20, 64)

    # 3. Normalisation spectrale identique à FeatureExtractor
    #    (mean/std par bande de fréquence, axis=1 = axe temporel)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / \
          (mel.std(axis=1, keepdims=True) + 1e-8)

    # 4. Zoom vers la taille attendue par le modèle
    #    Si MCU_SHAPE == MODEL_SHAPE → zoom=(1.0, 1.0), pas de modification
    if mel.shape != MODEL_SHAPE:
        zoom_factors = [MODEL_N_MEL / mel.shape[0], MODEL_N_FRAMES / mel.shape[1]]
        mel = scipy.ndimage.zoom(mel, zoom_factors, order=1)

    mel = mel[:MODEL_N_MEL, :MODEL_N_FRAMES]   # sécurité

    # 5. Tenseur Keras : (1, N_MEL, N_FRAMES, 1)
    tensor = mel[np.newaxis, ..., np.newaxis].astype(np.float32)

    return tensor, mel


# ── Soumission / sauvegarde ───────────────────────────────────
def submit_guess(guess):
    url = f"{HOSTNAME}/lelec210x/leaderboard/submit/{GROUP_KEY}/{guess}"
    try:
        response = requests.post(url, timeout=1.0)
        print(f"Statut : {response.status_code} {response.reason}", file=sys.stderr)
        if response.ok:
            print(f"Succès : '{guess}' enregistré.", file=sys.stderr)
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
        print(f"Guess '{guess}' sauvegardé dans {GUESS_FILE}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde : {e}", file=sys.stderr)
        return False


# ── Traitement d'une ligne UART ───────────────────────────────
def process_line(line):
    line = line.strip()
    if line:
        print(f"[DEBUG] reçu : {line[:80]}...", file=sys.stderr)

    if not line.startswith(PRINT_PREFIX):
        return

    hex_payload = line[len(PRINT_PREFIX):]
    try:
        raw_bytes = bytes.fromhex(hex_payload)
        data = np.frombuffer(raw_bytes, dtype=np.dtype('<u2'))

        expected_len = MCU_N_MELVECS * MCU_MELVEC_LENGTH
        print(f"  Taille reçue : {len(data)} valeurs (attendu {expected_len})", file=sys.stderr)

        if len(data) < expected_len:
            print(f"  ! Paquet trop court : {len(data)} < {expected_len} — ignoré", file=sys.stderr)
            return

        # Feature extraction + zoom automatique vers input_shape du modèle
        tensor, mel_2d = mcu_mel_to_tensor(data)

        probabilities = model.predict(tensor, verbose=0)[0]
        pred_idx      = np.argmax(probabilities)
        prediction    = classnames[pred_idx]
        confidence    = float(probabilities[pred_idx])

        print(f"\nSon détecté : {prediction} ({confidence:.2%})", file=sys.stderr)
        for c, p in zip(classnames, probabilities):
            bar = "█" * int(p * 20)
            print(f"  {c:<15}: {p:.4f}  {bar}", file=sys.stderr)

        if confidence < CONFIDENCE_THRESHOLD:
            print(
                f"  Confiance insuffisante ({confidence:.2%} < {CONFIDENCE_THRESHOLD:.0%}) — ignoré.",
                file=sys.stderr
            )
            return

        if AUTO_SUBMIT:
            submit_guess(prediction)
        else:
            save_guess_to_file(prediction, probabilities)

    except Exception as e:
        print(f"Erreur traitement : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("En attente de données depuis le pipe...", file=sys.stderr)
    for line in sys.stdin:
        process_line(line)


if __name__ == "__main__":
    main()