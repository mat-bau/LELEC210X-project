"""
uart-reader.py — acquisition only, no classification
Saves raw audio from UART to audio_acquisition/SUD11/
"""

import argparse
import sys
import numpy as np
import serial
import soundfile as sf
from pathlib import Path
from serial.tools import list_ports
from datetime import datetime

PRINT_PREFIX      = "SND:HEX:"
FREQ_SAMPLING_MCU = 10200
SAVE_DIR          = Path("audio_acquisition") / "SUD11"


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


def save_audio(raw_audio, counter):
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"acquisition_{timestamp}_{counter:04d}.wav"

    buf = np.asarray(raw_audio, dtype=np.float64)
    buf -= np.mean(buf)
    mx = np.max(np.abs(buf))
    if mx > 0:
        buf /= mx

    sf.write(path, buf, FREQ_SAMPLING_MCU)
    return path


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port serie (ex: /dev/cu.usbmodem...)")
    args = argParser.parse_args()

    if args.port is None:
        print("Ports disponibles :")
        for p in list(list_ports.comports()):
            print(f"  - {p.device}")
        print("\nUsage : uv run uart-reader.py -p PORT")
        sys.exit(0)

    print(f"Sauvegarde dans : {SAVE_DIR.resolve()}")

    counter = 0
    for raw_audio in reader(port=args.port):
        counter += 1
        path = save_audio(raw_audio, counter)
        print(f"  #{counter:04d}  {len(raw_audio)} samples  ->  {path.name}")