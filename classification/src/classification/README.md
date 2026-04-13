## 1 . Entraîner un nouveau modèle (CLI)
```sh
# Config de base (N_MEL=64, ReLU)
uv run python -m classification.training.train \
  --config base \
  --author mateo \
  --tags "baseline,relu"

# Config N_MEL=32 avec Swish
uv run python -m classification.training.train \
  --config resnet32 \
  --author mateo \
  --tags "32mel,swish,background-class"

# Surcharger le dossier de sortie
uv run python -m classification.training.train \
  --config resnet32 \
  --author mateo \
  --model-dir ./data/models/mateo_resnet32_test

```

Le modèle sera sauvegardé dans classification/data/models/{author}_{config}_{date}/ avec best_model.keras, config.json et model_config.pkl.

## 2. Lancer l'inférence temps-réel (UART)

Avec le nouveau pipeline (RealTimeVoter + background filtering)
```sh
uv run python -m classification.inference.classifier_pipe \
  --port /dev/cu.usbmodem... \
  --model-dir classification/data/models/models_resnet/valacc9821_test8625 \
  --url http://leaderboard:5000 \
  --key MA_CLE
```
Sans soumission (dry-run, pour tester)
```sh
uv run python -m classification.inference.classifier_pipe \
  --port /dev/cu.usbmodem... \
  --model-dir classification/data/models/models_resnet/valacc9821_test8625
```

En pipe depuis auth
```sh
uv run auth | uv run python -m classification.inference.classifier_pipe \
  --stdin \
  --model-dir classification/data/models/models_resnet/valacc9821_test8625 \
  --key MA_CLE
```
## 3. Comparer les activations (Tâche 4)
Dans un notebook ou script Python :
```python
import sys
sys.path.insert(0, 'classification/src')

from classification.configs import BaseConfig
from classification.datasets import Dataset
from classification.training.evaluate import compare_activations

dataset = Dataset()
for cls in ("background", "birds", "handsaw", "helicopter"):
    dataset.remove_class(cls)

cfg = BaseConfig()
results = compare_activations(
    dataset, cfg,
    activations=["relu", "gelu", "swish", "mish", "leaky_relu"],
    epochs=50,
)
print(results)  # {'relu': 0.92, 'swish': 0.94, ...}
```
## 4. Visualiser GradCAM (Tâche 5)
```py
import keras
import numpy as np
from classification.models.gradcam import plot_gradcam_grid, plot_gradcam_misclassified

model = keras.models.load_model("classification/data/models/models_resnet/valacc9821_test8625/best_model.keras")

# Grille complète
plot_gradcam_grid(model, X_test, y_test, classnames, n_examples=3,
                  save_path="gradcam_grid.png")

# Focus erreurs (fire/fireworks confusion)
plot_gradcam_misclassified(model, X_test, y_test, y_pred, classnames,
                           save_path="gradcam_errors.png")

```
## 5. Analyser les fausses alarmes (Tâche 3)
```py
from classification.training.evaluate import analyze_false_positive_temporal
from classification.data.dataset import load_background_samples

cfg.BACKGROUND_DATA_DIR = "path/to/background/wavs"
X_bg, _ = load_background_samples(cfg, extractor)
analyze_false_positive_temporal(model, X_bg, cfg, extractor)
# → statistiques + histogramme des confiances sur fenêtres de bruit
```
## 6. Prochaines étapes recommandées
Priorité | 	Action

Immédiat |	Ajouter config.json au modèle existant valacc9821_test8625 (pour que classifier_pipe.py puisse le charger)

Court terme	| Collecter des échantillons background forest → activer BACKGROUND_DATA_DIR → ré-entraîner

Expérimentation	| compare_activations swish vs relu sur 50 epochs pour voir si gain
Compétition |	Tester RealTimeVoter en conditions réelles avec les seuils par défaut (0.4/0.6)

Ajouter config.json au modèle existant (important)

Sans cette étape, classifier_pipe.py ne peut pas charger les anciens modèles proprement :
```py

import sys, pickle
sys.path.insert(0, 'classification/src')
from classification.configs import BaseConfig

with open("classification/data/models/models_resnet/valacc9821_test8625/model_config.pkl", "rb") as f:
    meta = pickle.load(f)

cfg = BaseConfig(
    N_MEL=meta["n_mel"],
    N_FFT=meta["n_fft"],
    HOP_LENGTH=meta["hop_length"],
    SAMPLE_RATE=meta["sample_rate"],
    MODEL_DIR="classification/data/models/models_resnet/valacc9821_test8625",
)
cfg.to_json("classification/data/models/models_resnet/valacc9821_test8625/config.json")
print("config.json créé")
```