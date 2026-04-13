# LELEC2102 — Classification pipeline refactor

## Contexte du projet

Pipeline temps-réel embarqué → PC :
STM32 (firmware C) → UART → uart-reader.py → ZMQ → classifier_pipe.py → ResNet CNN → label

Le firmware MCU capture de l'audio, calcule N_MELVECS=20 (à verif dans les fichiers core de mcu, si on peut ou pas les modif) vecteurs mel de longueur
MELVEC_LENGTH=20 (soit 400 uint16 par paquet), et les envoie sur UART avec le préfixe
"SND:HEX:". Le classifier_pipe.py reçoit ces 400 valeurs, les reshape en (20,20), puis
utilise scipy.ndimage.zoom pour les redimensionner à la shape attendue par le modèle
(N_MEL, N_FRAMES, 1) — donc le 87 en n_frames vient EXCLUSIVEMENT des paramètres DSP
Python (n_frames = int(DURATION_MS/1000 * SAMPLE_RATE / HOP_LENGTH) + 1 = 87 à 11025Hz (à vérifier si c'est ça ou 10200Hz)
hop=128 dur=1000ms) et NON du firmware. Le config.h MCU N'A PAS besoin d'être touché.

4 classes actives : chainsaw, fire, fireworks, gunshot.
Meilleurs résultats actuels : train=100%, val=97.6%, test=91.2% (TTA×5).
tu peux le voir dans le dossier models_resnet
Domain gap résiduel ~14% entre données synthétiques et micro embarqué réel.

Stack : Python 3.10, TensorFlow 2.18.1, Keras 3, uv, W&B (projet bau-mat-ucl/audio-classifier).
Hardware : Mac Studio M4 Max (Metal GPU), partagé avec équipiers sur Windows/Linux.

---

## Mission de l'agent

Tu es un agent senior ML/architecture. Tu dois exécuter les tâches suivantes dans l'ordre,
en validant chaque étape avant de passer à la suivante. Pour chaque fichier créé ou modifié,
explique brièvement le choix d'architecture.

---

## TÂCHE 1 — Restructuration du dossier classification/

Reorganise le dossier classification/ pour un projet multi-personnes avec hardwares différents. Format des commentaires : ca doit être fait comme par un humain donc tout ce qui est possible d'écrire avec un clavier QWERTY, pas d'émoji pas de lettre grecque ou romaine etc. Pour le training garde wandb pour la visualisation de l'évolution de l'entrainement.

Structure cible :
classification/
├── configs/
│   ├── base_config.py          # Config parent avec tous les paramètres commentés
│   ├── resnet32.py             # Config N_MEL=32 (nouvelle cible)
│   ├── resnet64.py             # Config N_MEL=64 (actuelle, garder pour comparaison)
│   └── README.md               # Comment créer sa propre config
├── models/
│   ├── resnet.py               # Architecture ResNet (refactored depuis resnet_classifier.py)
│   ├── gradcam.py              # GradCAM standalone
│   └── activations.py         # Fonctions d'activation alternatives testables
├── data/
│   ├── augmentation.py         # AudioAugmentation
│   ├── dataset.py              # prepare_dataset, MelExtractor, loaders
│   └── mixup.py               # MixupGenerator
├── training/
│   ├── train.py               # run_pipeline() entry point
│   ├── callbacks.py           # GracefulStop, _build_callbacks
│   └── evaluate.py            # evaluate_model, plot_history
├── inference/
│   ├── classifier_pipe.py     # (déplacé depuis mcu/) adapté pour charger model_config.pkl
│   └── predict.py             # predict_with_tta, interface CLI
├── saved_models/
│   └── README.md              # Convention de nommage
└── notebooks/
    └── exploration.ipynb      # Notebook de départ propre

Règles de nommage des modèles sauvegardés (dans saved_models/) :
  {auteur}_{config_name}_{date}_{val_acc}_{test_acc}/
    ├── model.keras
    ├── model.weights.h5
    └── config.json             ← NOUVEAU : json lisible humain avec TOUS les params

Le config.json doit contenir :
{
  "author": "mateo",
  "hardware": "M4Max-Metal",
  "config_name": "resnet32",
  "date": "2025-04-13",
  "classnames": ["chainsaw", "fire", "fireworks", "gunshot"],
  "input_shape": [32, 87, 1],
  "n_mel": 32,
  "n_fft": 512,
  "hop_length": 128,
  "sample_rate": 11025,
  "duration_ms": 1000,
  "dropout_rate": 0.3,
  "mixup_alpha": 0.4,
  "label_smoothing": 0.1,
  "batch_size": 256,
  "val_accuracy": 0.982,
  "test_accuracy": 0.862,
  "tta_steps": 5,
  "keras_version": "3.x.x",
  "notes": "Baseline avec background class"
}

Cela remplace le model_config.pkl actuel (garder compatibilité pickle pour la transition).
classifier_pipe.py doit pouvoir charger les deux formats (json prioritaire, pkl fallback).

---

## TÂCHE 2 — Nouvelle config N_MEL=32 et compréhension du n_frames

Crée configs/resnet32.py en héritant de base_config.py avec :
- N_MEL = 32
- DROPOUT_RATE = 0.3   (val=98% indique bonne généralisation, moins de régularisation)
- MIXUP_ALPHA = 0.4    (améliorer frontière fire/fireworks)
- COSINE_RESTART_EPOCHS = 40

Dans configs/base_config.py, ajoute une méthode compute_n_frames() et une propriété
n_frames qui calcule automatiquement :
  n_frames = int(DURATION_MS / 1000 * SAMPLE_RATE / HOP_LENGTH) + 1

et un commentaire explicite :
  # n_frames = 87 avec SR=11025, HOP=128, DUR=1000ms
  # Ce nombre est INDÉPENDANT du firmware MCU (N_MELVECS=20, MELVEC_LENGTH=20).
  # classifier_pipe.py zoome les 20×20 MCU vers (N_MEL, n_frames) avec scipy.
  # NE PAS toucher config.h pour changer N_MEL ou n_frames.

---

## TÂCHE 3 — Background class et stratégie de classification temps-réel

### Problème
En conditions réelles, le MCU envoie ~3 paquets pendant la durée d'un événement sonore.
Si le vrai son est classifié au paquet 1 mais que les paquets 2 et 3 sont du "bruit de fond
post-événement", on envoie 3 guesses dont 2 potentiellement faux, ce qui nuit au score.

### Demande
Analyse ce problème et implémente la stratégie suivante dans inference/classifier_pipe.py :

**Stratégie 1 — Background class + seuil de confiance** :
- Ajouter "background" comme 5ème classe dans l'entraînement
- Ne soumettre un guess que si P(background) < CONFIDENCE_THRESHOLD (ex: 0.4)
- Et si max(P(autres_classes)) > CONFIDENCE_THRESHOLD (ex: 0.6)
- Implémenter un mode "vote majoritaire" sur une fenêtre glissante de 3 paquets :
    seuls les guesses non-background sont comptés, le label le plus fréquent est soumis

**Stratégie 2 — Fenêtre temporelle élargie** (alternative à évaluer) :
- Accumuler N_PACKETS=3 paquets consécutifs avant de classer
- Concaténer les spectrogrammes sur l'axe temporel (→ input 3× plus large)
- Classer une seule fois sur la fenêtre complète
- Implémente un flag USE_SLIDING_WINDOW dans la config pour switcher

Dans training/evaluate.py, ajoute une section "analyse des faux positifs temporels" qui
simule ce comportement sur le test set et compare les deux stratégies.

Pour le dataset background :
- Ajouter BACKGROUND_DATA_DIR dans base_config.py
- Créer data/dataset.py une fonction load_background_samples() qui charge des fichiers
  .wav depuis ce dossier avec AUG_PASSES=0 (le bruit de fond réel ne doit pas être augmenté
  avec du pitch shift etc.)
- Documenter dans un README.md dans le dossier des données comment enregistrer du background
  utile (30s dans la salle de TP, distance micro ~50cm, sans événement)

---

## TÂCHE 4 — Fonctions d'activation alternatives dans models/activations.py

Crée models/activations.py avec les variantes suivantes à tester via un flag dans la config
(ACTIVATION = "relu" | "gelu" | "swish" | "mish" | "leaky_relu").

Pour chaque activation, implémente-la compatible Keras 3 et documente dans un commentaire
pourquoi elle pourrait améliorer notre cas :

- **ReLU** (baseline) : rapide, bien connu, risque dying ReLU
- **GELU** (Gaussian Error Linear Unit) : utilisée dans les transformers, lissage du seuil
  d'activation, peut améliorer la généralisation sur les sons "intermédiaires"
- **Swish** (x·σ(x)) : non-monotone, conserve une partie de l'information négative,
  montré supérieur à ReLU sur des CNNs profonds (Google Brain 2017)
- **Mish** (x·tanh(softplus(x))) : encore plus lisse que Swish, meilleurs résultats sur
  de petits datasets bruités selon la littérature
- **LeakyReLU** (α=0.01) : évite dying ReLU, simple, bon point de départ

Dans models/resnet.py, le residual_block() doit accepter un paramètre activation_fn
qui est passé depuis la config. Le build_model() passe automatiquement la bonne activation.

Ajoute dans training/evaluate.py une fonction compare_activations() qui lance un mini-run
de 50 epochs pour chaque activation et trace val_accuracy vs epoch pour comparer.

---

## TÂCHE 5 — GradCAM dans models/gradcam.py

Implémente le code GradCAM complet et standalone, utilisable en post-entraînement sans
ré-entraîner le modèle.

```python
# Interface publique de gradcam.py

def compute_gradcam(
    model: keras.Model,
    spectrogram: np.ndarray,   # shape (1, N_MEL, n_frames, 1)
    class_idx: int,
    last_conv_layer_name: str = None,  # None = autodetect (dernière couche ReLU avant GAP)
) -> np.ndarray:
    """Retourne heatmap normalisée (0-1) de shape (N_MEL, n_frames)."""

def auto_detect_last_conv_layer(model: keras.Model) -> str:
    """Trouve automatiquement le nom de la dernière couche conv/relu avant le GAP."""

def plot_gradcam_grid(
    model: keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    classnames: list,
    n_examples: int = 3,
    save_path: str = None,
) -> None:
    """
    Grid n_classes × n_examples×2 (original + heatmap superposée).
    Sauvegarde dans saved_models/{run_name}/gradcam.png si save_path fourni.
    """

def plot_gradcam_misclassified(
    model: keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classnames: list,
    save_path: str = None,
) -> None:
    """
    Focus sur les erreurs : affiche les cas où fireworks est confondu avec gunshot
    et vice-versa. Utile pour le rapport.
    """
```

Pour l'autodetection de la couche, cherche la dernière couche dont le nom contient "re_lu"
ou "relu" ET dont l'output shape est 4D (batch, h, w, channels) — juste avant le
GlobalAveragePooling2D.

---

## TÂCHE 6 — Intégration finale et tests

1. Crée training/train.py comme entry point CLI :
python3 -m classification.training.train 
--config resnet32 
--author mateo 
--tags "background-class,32mel,swish"
2. Vérifie que classifier_pipe.py charge correctement un modèle sauvegardé au nouveau format
   (config.json) et fonctionne avec le pipeline ZMQ existant.

3. Crée un fichier QUICKSTART.md à la racine de classification/ :
   - Comment lancer un entraînement
   - Comment ajouter sa propre config hardware
   - Convention de nommage des sauvegardes
   - Comment interpréter les GradCAM pour le rapport

4. Vérifie que `uv sync` fonctionne sur la config toml actuelle (tensorflow-metal marqué
   sys_platform == 'darwin') avant de terminer.

---

## Contraintes techniques

- Tout le code doit être compatible Python 3.10, TF 2.18.1, Keras 3
- Aucun import `from tensorflow.keras import ...` — utiliser `from keras import ...`
- Aucun caractère non-ASCII dans le code (commentaires en anglais ou français sans accents)
- Les fonctions ont une seule responsabilite (single-purpose)
- Pas de global mutable — tout passe par la Config dataclass
- W&B : chaque run log son config.json comme artifact en plus du modele
- Le code doit tourner identiquement sur Mac M4 Max (Metal) et Linux/Windows (CPU ou CUDA)