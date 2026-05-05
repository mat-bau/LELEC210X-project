# Classification Module

Audio classification pipeline for the LELEC210X embedded system project.

**System pipeline:**
MCU (STM32L4A6ZG) → S2LP radio (872 MHz) → LimeSDR receiver → GNU Radio (ZMQ) → auth → **classification** → leaderboard

## Installation

```bash
cd classification
uv sync
uv run python -m ipykernel install --user --name LELEC210X
```

> On some platforms you may need extra packages to play audio —
> see [issue #27](https://github.com/LELEC210X/LELEC210X/issues/27) if errors occur.

Run notebooks:

```bash
uv run jupyter notebook
# on WSL (no browser):
uv run jupyter notebook --no-browser
```

---

## Commands reference

### 1. Training

```bash
# Default config (N_MEL=64, sr=11025)
uv run python -m classification.training.train --config base --author alice

# MCU-aligned config (N_MEL=20, sr=10200, HOP=512 — RECOMMENDED, zero zoom at inference)
uv run python -m classification.training.train --config mcu_match --author alice

# See all options
uv run python -m classification.training.train --help
```

**Available configs:**

| Config | N_MEL | n_frames | SR | HOP | Notes |
|---|---|---|---|---|---|
| `base` | 64 | 87 | 11025 | 128 | Default, high resolution |
| `resnet32` | 32 | 87 | 11025 | 128 | Smaller, less domain gap |
| `mcu_match` | 20 | 32 | 10200 | 512 | **Exact MCU alignment — recommended** |
| `mel24_mcu` | 24 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=24 |
| `mel28_mcu` | 28 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=28 |
| `mel32_mcu` | 32 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=32 |

**MCU alignment:** The firmware sends packets of shape `(MELVEC_LENGTH=20, N_MELVECS=32)`.
With `mcu_match`, the model trains on exactly this shape — no `scipy.ndimage.zoom`
at inference, no interpolation artifacts, no domain gap on spatial dimensions.

**Optional config flags:**

```python
cfg.USE_SE_BLOCKS = True        # Squeeze-and-Excitation channel attention (+1-2% acc)
cfg.MIXED_PRECISION = True      # fp16 training (faster on GPU/MPS)
cfg.NORMALIZATION_STATS_PATH = "auto"  # global per-band normalisation
```

---

### 2. N_MEL sweep experiment

Compare accuracy vs. MCU packet size for N_MEL ∈ {20, 24, 28, 32}.
All variants use MCU-aligned DSP parameters (sr=10200, HOP=512).

```bash
# Full sweep — trains 4 models sequentially, compares them
uv run sweep-mel --epochs 100 --output-dir ./mel_sweep_results

# Quick smoke test
uv run sweep-mel --epochs 5 --output-dir /tmp/sweep_smoke

# Custom subset
uv run sweep-mel --n-mel 20 --n-mel 32 --epochs 100

uv run sweep-mel --help
```

**LimeSDR packet budget (N_MEL × 32 frames × 2 bytes):**

| N_MEL | Payload | Notes |
|---|---|---|
| 20 | 1280 B | Current firmware, upsampling=8 OK |
| 24 | 1536 B | Check RF quality with team |
| 28 | 1792 B | Check RF quality with team |
| 32 | 2048 B | May need Lime upsampling reduction (8→4) |

> When changing `MELVEC_LENGTH` in the firmware you must **re-record all classes**.
> The mel computation changes and existing recordings become incompatible.

**Outputs in `./mel_sweep_results/`:**
- `sweep_results.json` — val/test accuracy per N_MEL
- `mel_sweep_comparison.png` — 3-panel chart
- `sweep_mel20/`, `sweep_mel24/`, … — individual model dirs

---

### 3. Standalone test-set evaluation

Evaluate any saved model without retraining:

```bash
# Full evaluation with GradCAM
uv run evaluate-model --model-dir ./data/models/my_run

# Fast mode (skip GradCAM)
uv run evaluate-model --model-dir ./data/models/my_run --no-gradcam

# Override data directory
uv run evaluate-model --model-dir ./data/models/my_run \
    --real-data /path/to/audio_files

uv run evaluate-model --help
```

**Outputs in `{model_dir}/evaluation/`:**
- `confusion_matrix.png` — test set confusion matrix (with TTA)
- `per_class_metrics.png` — precision / recall / F1 per class
- `metrics.json` — all metrics as JSON
- `gradcam_grid.png` — GradCAM heatmaps per class (3 examples each)
- `gradcam_misclassified.png` — GradCAM on misclassified samples

---

### 4. Audio-JEPA pretraining (LeWorldModel)

Self-supervised pretraining based on Yann LeCun's JEPA
("A Path Towards Autonomous Machine Intelligence", 2022).
Trains without labels — learns to predict hidden spectrogram patch embeddings
from visible context. Improves generalisation and reduces domain gap.

```bash
# Pretrain on all audio (labels are ignored — use any .wav directory)
uv run pretrain-jepa \
    --data-dir ./mcu/hands_on_audio_acquisition/audio_files \
    --config mcu_match \
    --output-dir ./jepa_pretrained \
    --epochs 150

# Smoke test (5 epochs)
uv run pretrain-jepa --data-dir ./audio_files --epochs 5

uv run pretrain-jepa --help
```

**Outputs in `./jepa_pretrained/`:**
- `jepa_encoder.weights.h5` — pretrained ResNet encoder weights
- `jepa_loss.png` — pretraining loss curve

**Fine-tune after pretraining:**

```python
from classification.models.jepa import fine_tune_from_jepa
from classification.configs.mcu_match import MCUMatchConfig

model = fine_tune_from_jepa(
    encoder_weights_path="./jepa_pretrained/jepa_encoder.weights.h5",
    input_shape=(20, 32, 1),   # must match pretraining config
    n_classes=4,
    cfg=MCUMatchConfig(),
    freeze_epochs=10,          # encoder frozen for first 10 epochs
)
# then use model.fit() or run_pipeline() normally
```

---

### 5. Real-time pipeline (GNURadio → classify)

Full pipeline via auth + classify:

```bash
uv run auth | uv run classify \
    --model-dir ./data/models/my_run \
    --url http://leaderboard:5000 \
    --key MY_API_KEY
```

Direct UART (bypasses GNURadio, for MCU bench testing):

```bash
uv run python -m classification.inference.classifier_pipe \
    --port /dev/cu.usbmodem1234 \
    --model-dir ./data/models/my_run \
    --url http://leaderboard:5000 \
    --key MY_API_KEY
```

Local display only (no leaderboard):

```bash
uv run python -m classification.inference.classifier_pipe \
    --port /dev/cu.usbmodem1234 \
    --model-dir ./data/models/my_run
```

---

### 6. GradCAM visualisation (Python API)

```python
import keras
from classification.models.gradcam import (
    plot_gradcam_grid, plot_gradcam_misclassified
)

model = keras.models.load_model("./data/models/my_run/best_model.keras")

# Grid: one row per class, 3 examples
plot_gradcam_grid(model, X_test, y_test, classnames,
                  n_examples=3, save_path="./gradcam_grid.png")

# Misclassified samples
plot_gradcam_misclassified(model, X_test, y_true, y_pred, classnames,
                           save_path="./gradcam_errors.png")
```

---

### 7. Audio splitting

Split a long audio file into short clips for the training dataset:

```bash
uv run split-audio my_recording.wav \
    --num-pieces 50 \
    --duration 5.0 \
    --directory ./mcu/hands_on_audio_acquisition/audio_files/chainsaw/training \
    --prefix chainsaw

uv run split-audio --help
```

**Expected dataset structure:**

```
audio_files/
├── chainsaw/
│   ├── training/   ← .wav files for training
│   └── test/       ← .wav files for evaluation (isolated from training)
├── fire/
├── fireworks/
└── gunshot/
```

---

## Configuration reference

Key fields of `BaseConfig`:

| Field | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 11025 | Hz. Use 10200 for MCU alignment. |
| `N_MEL` | 64 | Mel bands. Must match firmware `MELVEC_LENGTH`. |
| `N_FFT` | 512 | FFT window size. |
| `HOP_LENGTH` | 128 | FFT hop. Use 512 for MCU alignment (no overlap). |
| `DURATION_MS` | 1000 | Analysis window. Use 1557 for MCU alignment (→ 32 frames). |
| `REAL_DATA_DIR` | `…/audio_files` | Path to labelled audio dataset. |
| `REAL_AUG_PASSES` | 60 | Augmentation passes per training file. |
| `EPOCHS` | 300 | Max training epochs. |
| `USE_SE_BLOCKS` | False | Squeeze-and-Excitation channel attention. |
| `MIXED_PRECISION` | False | fp16 training (GPU/MPS speedup). |
| `NORMALIZATION_STATS_PATH` | `""` | Path to `norm_stats.npz` for global normalisation. |
| `CONFIDENCE_THRESHOLD` | 0.6 | Min confidence to submit in real-time mode. |
| `BACKGROUND_THRESHOLD` | 0.4 | Max P(background) before discarding packet. |
| `VOTER_WINDOW_SIZE` | 3 | Majority vote window size. |
| `WANDB_PROJECT` | `audio-classifier` | Weights & Biases project name. |

---

## Adding training data

### From YouTube

```bash
uv run youtube-dl -x --audio-format=wav "<youtube-url>"
uv run split-audio downloaded.wav --num-pieces 50 --duration 5 \
    --directory ./audio_files/chainsaw/training
```

See the [TA YouTube playlist](https://youtube.com/playlist?list=PLK2PsMuicSN8Y7ovsXjypFADW5EeGVn36).

### From the MCU (UART acquisition)

```bash
python mcu/hands_on_audio_acquisition/uart-reader.py
```

Sort recordings into `audio_files/{class}/training/` and `audio_files/{class}/test/`.
