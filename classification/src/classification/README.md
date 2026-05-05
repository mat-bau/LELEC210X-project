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

> On some platforms you may need extra packages to play audio.
> See [issue #27](https://github.com/LELEC210X/LELEC210X/issues/27) if this occurs.

---

## Commands reference

### 1. Training

```bash
# Train with default config (N_MEL=64, sr=11025)
uv run python -m classification.training.train --config base --author name

# Train MCU-aligned config (N_MEL=20, sr=10200, HOP=512 → zero zoom at inference)
uv run python -m classification.training.train --config mcu_match --author name

# Train with smaller N_MEL (faster, less domain gap)
uv run python -m classification.training.train --config resnet32 --author name

# Full options
uv run python -m classification.training.train --help
```

Available configs:

| Config | N_MEL | n_frames | SR | HOP | Notes |
|---|---|---|---|---|---|
| `base` | 64 | 87 | 11025 | 128 | Default, high resolution |
| `resnet32` | 32 | 87 | 11025 | 128 | Faster, less domain gap |
| `mcu_match` | 20 | 32 | 10200 | 512 | **Exact MCU alignment — recommended** |
| `mel24_mcu` | 24 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=24 |
| `mel28_mcu` | 28 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=28 |
| `mel32_mcu` | 32 | 32 | 10200 | 512 | Requires firmware MELVEC_LENGTH=32 |

**MCU alignment explained:**
The MCU firmware sends packets of shape `(MELVEC_LENGTH=20, N_MELVECS=32)`.
With `mcu_match`, the model is trained on exactly this shape — no `scipy.ndimage.zoom`
is needed at inference, eliminating interpolation artifacts and domain gap.

**Optional features (in config):**
```python
cfg.USE_SE_BLOCKS = True       # Squeeze-and-Excitation attention (+1-2% acc)
cfg.MIXED_PRECISION = True     # fp16 training (faster on GPU/MPS)
cfg.NORMALIZATION_STATS_PATH = "auto"  # global per-band normalisation
```

### 2. N_MEL sweep experiment

Compare classification accuracy vs. MCU packet size for N_MEL ∈ {20, 24, 28, 32}:

```bash
# Full sweep (trains 4 models sequentially)
uv run sweep-mel --epochs 100 --output-dir ./mel_sweep_results

# Quick smoke test (5 epochs each)
uv run sweep-mel --epochs 5 --output-dir /tmp/sweep_smoke

# Custom subset
uv run sweep-mel --n-mel 20 --n-mel 32 --epochs 100

uv run sweep-mel --help
```

**Output in `./mel_sweep_results/`:**
- `sweep_results.json` — val/test accuracy per N_MEL
- `mel_sweep_comparison.png` — 3-panel chart: val acc / test acc / packet size
- `sweep_mel20/`, `sweep_mel24/`, … — individual model directories

**LimeSDR packet budget:**

| N_MEL | Payload | Notes |
|---|---|---|
| 20 | 1280 B | Current firmware, upsampling=8 OK |
| 24 | 1536 B | Check RF quality |
| 28 | 1792 B | Check RF quality |
| 32 | 2048 B | May require upsampling reduction (8→4 or 8→2) |

### 3. Standalone test-set evaluation

Evaluate any saved model on the test set (no retraining needed):

```bash
# Full evaluation with GradCAM
uv run evaluate-model --model-dir ./data/models/my_run

# Fast mode (skip GradCAM, useful for quick accuracy check)
uv run evaluate-model --model-dir ./data/models/my_run --no-gradcam

# Override data directory
uv run evaluate-model --model-dir ./data/models/my_run \
    --real-data /path/to/audio_files

uv run evaluate-model --help
```

**Output in `{model_dir}/evaluation/`:**
- `confusion_matrix.png` — test set confusion matrix with TTA
- `per_class_metrics.png` — precision / recall / F1 per class
- `metrics.json` — all metrics as JSON
- `gradcam_grid.png` — GradCAM heatmaps per class (3 examples each)
- `gradcam_misclassified.png` — GradCAM on misclassified samples

### 4. Audio-JEPA pretraining (LeWorldModel)

Self-supervised pretraining based on Yann LeCun's JEPA architecture
("A Path Towards Autonomous Machine Intelligence", 2022).

Trains without labels: predicts hidden spectrogram patch embeddings from
visible context. Improves generalisability, especially useful when labelled
data is scarce or when bridging the MCU/Python domain gap.

```bash
# Pretrain on all available audio (labelled or not — labels are ignored)
uv run pretrain-jepa \
    --data-dir ./mcu/hands_on_audio_acquisition/audio_files \
    --config mcu_match \
    --output-dir ./jepa_pretrained \
    --epochs 150

# Smoke test
uv run pretrain-jepa --data-dir ./audio_files --epochs 5 --output-dir /tmp/jepa_test

uv run pretrain-jepa --help
```

**Output in `./jepa_pretrained/`:**
- `jepa_encoder.weights.h5` — pretrained ResNet encoder weights
- `jepa_loss.png` — pretraining loss curve

**Fine-tuning after pretraining:**
```python
from classification.models.jepa import fine_tune_from_jepa
from classification.configs.mcu_match import MCUMatchConfig

cfg = MCUMatchConfig()
model = fine_tune_from_jepa(
    encoder_weights_path="./jepa_pretrained/jepa_encoder.weights.h5",
    input_shape=(20, 32, 1),    # must match pretraining config
    n_classes=4,
    cfg=cfg,
    freeze_epochs=10,           # freeze encoder for first 10 epochs
)
# Then train normally with run_pipeline() or model.fit()
```

### 5. Real-time pipeline (GNURadio → classify)

Full pipeline: auth module + classify piped together:

```bash
uv run auth | uv run classify \
    --model-dir ./data/models/my_run \
    --url http://leaderboard:5000 \
    --key MY_API_KEY
```

Direct UART connection (bypasses GNURadio, for MCU testing):

```bash
uv run python -m classification.inference.classifier_pipe \
    --port /dev/cu.usbmodem1234 \
    --model-dir ./data/models/my_run \
    --url http://leaderboard:5000 \
    --key MY_API_KEY
```

Without leaderboard submission (local display only):

```bash
uv run python -m classification.inference.classifier_pipe \
    --port /dev/cu.usbmodem1234 \
    --model-dir ./data/models/my_run \
    --no-plot
```

### 6. GradCAM visualisation (Python)

```python
import numpy as np
import keras
from classification.models.gradcam import (
    plot_gradcam_grid, plot_gradcam_misclassified, auto_detect_last_conv_layer
)

# Load model and test data
model = keras.models.load_model("./data/models/my_run/best_model.keras")

# Plot GradCAM grid (one row per class, 3 examples)
plot_gradcam_grid(model, X_test, y_test, classnames,
                  n_examples=3,
                  save_path="./gradcam_grid.png")

# Misclassified samples only
plot_gradcam_misclassified(model, X_test, y_true, y_pred, classnames,
                           save_path="./gradcam_errors.png")

# Single sample
from classification.models.gradcam import compute_gradcam
layer_name = auto_detect_last_conv_layer(model)
heatmap = compute_gradcam(model, X_test[0], class_idx=2, last_conv_layer_name=layer_name)
```

### 7. Audio splitting utility

Split a long audio file into many short clips for your dataset:

```bash
# Split into 50 clips of 5s each, saved to the chainsaw/training folder
uv run split-audio my_recording.wav \
    --num-pieces 50 \
    --duration 5.0 \
    --directory ./mcu/hands_on_audio_acquisition/audio_files/chainsaw/training \
    --prefix chainsaw

uv run split-audio --help
```

Expected dataset folder structure:
```
audio_files/
├── chainsaw/
│   ├── training/   ← wav files used for training
│   └── test/       ← wav files used for evaluation (isolated)
├── fire/
│   ├── training/
│   └── test/
├── fireworks/
│   ├── training/
│   └── test/
└── gunshot/
    ├── training/
    └── test/
```

---

## Configuration reference

All config fields with defaults from `BaseConfig`:

| Field | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 11025 | Audio sample rate (Hz). Set 10200 for MCU alignment. |
| `N_MEL` | 64 | Mel bands. Must match firmware `MELVEC_LENGTH`. |
| `N_FFT` | 512 | FFT window size. |
| `HOP_LENGTH` | 128 | FFT hop (512 for MCU alignment = no overlap). |
| `DURATION_MS` | 1000 | Analysis window duration. |
| `REAL_DATA_DIR` | `../../mcu/…/audio_files` | Path to labelled audio dataset. |
| `REAL_AUG_PASSES` | 60 | Augmentation passes per real audio file. |
| `BATCH_SIZE` | 256 | Training batch size. |
| `EPOCHS` | 300 | Max training epochs (early stopping active). |
| `LEARNING_RATE` | 5e-4 | Adam learning rate. |
| `EARLY_STOPPING_PATIENCE` | 60 | Epochs without improvement before stopping. |
| `USE_SE_BLOCKS` | False | Squeeze-and-Excitation channel attention. |
| `MIXED_PRECISION` | False | fp16 training (enable for GPU/MPS speed). |
| `NORMALIZATION_STATS_PATH` | `""` | Path to `norm_stats.npz` for global normalisation. |
| `CONFIDENCE_THRESHOLD` | 0.6 | Real-time: min confidence before submitting. |
| `BACKGROUND_THRESHOLD` | 0.4 | Real-time: max P(background) before discarding. |
| `VOTER_WINDOW_SIZE` | 3 | Real-time: majority vote window size. |
| `MODEL_DIR` | `./data/models/…` | Output directory for saved model. |
| `WANDB_PROJECT` | `audio-classifier` | Weights & Biases project name. |

---

## Notebooks

```bash
uv run jupyter notebook
```

Available notebooks:
- `h1_feature_vectors.ipynb` — feature extraction basics
- `h5a1_toy.ipynb` — toy classification problem
- `h5a2_audio.ipynb` — audio processing tutorial
- `h5a3_confidence_and_memory.ipynb` — confidence thresholds + memory analysis

---

## Adding training data

### Extend with YouTube audio

```bash
uv run youtube-dl -x --audio-format=wav "<youtube-url>"
uv run split-audio downloaded.wav --num-pieces 50 --duration 5 \
    --directory ./audio_files/chainsaw/training
```

See the [TA YouTube playlist](https://youtube.com/playlist?list=PLK2PsMuicSN8Y7ovsXjypFADW5EeGVn36) for pre-curated content.

### Record your own dataset with the MCU

1. Connect MCU via UART.
2. Run the UART reader to record mel spectrograms as WAV files:
   ```bash
   python mcu/hands_on_audio_acquisition/uart-reader.py
   ```
3. Sort recordings into `audio_files/{class}/training/` and `audio_files/{class}/test/`.

> **When changing `MELVEC_LENGTH` in the MCU firmware**, all existing recordings
> become invalid (different mel computation). You must re-record all classes.
> Also verify the LimeSDR packet budget — larger N_MEL means larger payloads
> which may require reducing the Lime upsampling factor.
