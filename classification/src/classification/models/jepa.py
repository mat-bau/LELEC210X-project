"""
Audio-JEPA: Joint Embedding Predictive Architecture for mel spectrograms.

Based on: "A Path Towards Autonomous Machine Intelligence" — Yann LeCun, 2022.
Architecture variant: I-JEPA (Image JEPA) adapted for 2-D mel spectrograms.

Key idea (LeCun's insight):
    Predict REPRESENTATIONS of masked regions (not raw pixels/spectrogram values).
    Loss in latent space → model learns structured, semantic features rather than
    low-level texture details.

Architecture:
    Context encoder  : ResNet body (no GAP/Dense) — processes visible patches.
    Target encoder   : EMA copy of context encoder — no gradient, updated by EMA.
    Predictor        : 2-layer MLP — predicts target embeddings at masked positions.

For MCUMatchConfig input (20, 32, 1):
    After encoder: feature map (3, 4, 256) = 12 spatial positions.
    Mask ratio 0.5 → 6 positions masked → predictor predicts their embeddings.

Usage — pretraining:
    uv run pretrain-jepa \\
        --data-dir ./audio_unlabeled \\
        --output-dir ./jepa_pretrained \\
        --config mcu_match \\
        --epochs 150

Usage — fine-tuning after pretraining:
    from classification.models.jepa import fine_tune_from_jepa
    model = fine_tune_from_jepa(
        encoder_weights_path="./jepa_pretrained/jepa_encoder.weights.h5",
        input_shape=(20, 32, 1),
        n_classes=4,
        cfg=MCUMatchConfig(),
    )
    # then pass model to your normal training loop
"""
from __future__ import annotations

import os

import click
import numpy as np
import keras
from keras import layers, models, regularizers
import tensorflow as tf

from ..configs.base_config import BaseConfig
from .resnet import build_encoder_body, build_model
from .activations import get_activation_fn


# ---------------------------------------------------------------------------
# Masking helpers (TF-native — works in both eager and graph mode)
# ---------------------------------------------------------------------------

def _make_mask_tf(n_positions: int, mask_ratio: float,
                  batch_size: tf.Tensor) -> tf.Tensor:
    """
    Random binary mask using TF operations (graph-mode compatible).

    :returns: bool tensor of shape (batch_size, n_positions)
              True = masked (hidden from context encoder)
    """
    noise = tf.random.uniform((batch_size, n_positions))
    # Top mask_ratio positions are masked (True)
    threshold = tf.cast(mask_ratio, tf.float32)
    return noise < threshold   # (B, n_pos) bool


def _apply_feature_mask(features: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Zero out masked positions in the feature map.

    :param features: (batch, H, W, C) feature tensor
    :param mask:     (batch, H*W) bool tensor — True = zero out
    :returns:        (batch, H, W, C) with masked positions zeroed
    """
    shape = tf.shape(features)
    B, H, W, C = shape[0], features.shape[1], features.shape[2], features.shape[3]
    n_pos = H * W
    flat = tf.reshape(features, (B, n_pos, C))          # (B, n_pos, C)
    mask_f = tf.cast(tf.logical_not(mask), tf.float32)  # 1=visible, 0=masked
    mask_f = tf.expand_dims(mask_f, axis=-1)             # (B, n_pos, 1)
    flat = flat * mask_f
    return tf.reshape(flat, tf.stack([B, H, W, C]))


# ---------------------------------------------------------------------------
# Predictor network
# ---------------------------------------------------------------------------

def _build_predictor(n_positions: int, embed_dim: int = 256,
                     hidden_dim: int = 512,
                     pos_dim: int = 32) -> keras.Model:
    """
    Simple per-position MLP predictor.

    Input: (batch, n_positions, embed_dim + pos_dim) — context embeddings +
           positional embeddings (masked positions are zero in embed part).
    Output: (batch, n_positions, embed_dim) — predicted target embeddings.

    Shared weights across positions (TimeDistributed).
    """
    inp = layers.Input(shape=(n_positions, embed_dim + pos_dim), name="pred_input")
    x = layers.TimeDistributed(
        layers.Dense(hidden_dim, activation="gelu"), name="pred_hidden"
    )(inp)
    x = layers.TimeDistributed(
        layers.LayerNormalization(), name="pred_ln"
    )(x)
    x = layers.TimeDistributed(
        layers.Dense(embed_dim), name="pred_output"
    )(x)
    return models.Model(inp, x, name="JEPA_Predictor")


# ---------------------------------------------------------------------------
# JEPAPretrainer — Keras subclassed model
# ---------------------------------------------------------------------------

class JEPAPretrainer(keras.Model):
    """
    JEPA pretraining model for mel spectrograms.

    call() is NOT used for training — override train_step() instead.

    Attributes:
        context_encoder : ResNet body (trained via gradients)
        target_encoder  : EMA copy of context_encoder (no gradient)
        predictor       : MLP that predicts target embeddings at masked positions
        pos_embeddings  : learnable positional embeddings (n_positions, pos_dim)
    """

    def __init__(self, input_shape: tuple, cfg: BaseConfig,
                 mask_ratio: float = 0.5,
                 ema_decay: float = 0.996,
                 hidden_dim: int = 512,
                 pos_dim: int = 32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay  = ema_decay
        self._input_shape = input_shape

        # Build both encoders from the same architecture
        self.context_encoder = build_encoder_body(input_shape, cfg)
        self.target_encoder  = build_encoder_body(input_shape, cfg)

        # Compute n_positions from a dummy forward pass
        dummy = tf.zeros((1,) + input_shape)
        feat_map = self.context_encoder(dummy, training=False)
        self._feat_H = int(feat_map.shape[1])
        self._feat_W = int(feat_map.shape[2])
        self._embed_dim = int(feat_map.shape[3])
        self._n_positions = self._feat_H * self._feat_W

        # Positional embeddings: one per spatial position
        self.pos_embeddings = self.add_weight(
            name="pos_embeddings",
            shape=(self._n_positions, pos_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Predictor
        self.predictor = _build_predictor(
            self._n_positions, self._embed_dim, hidden_dim, pos_dim)

        # Initialise target encoder with context encoder weights
        self.target_encoder.set_weights(self.context_encoder.get_weights())
        # Freeze target encoder — updated only via EMA
        self.target_encoder.trainable = False

    def _ema_update(self) -> None:
        """Update target encoder weights: θ_t ← α·θ_t + (1-α)·θ_c"""
        for tw, cw in zip(self.target_encoder.weights,
                          self.context_encoder.weights):
            tw.assign(self.ema_decay * tw + (1.0 - self.ema_decay) * cw)

    def train_step(self, data):
        # data is (X,) or (X, _) — labels not needed
        if isinstance(data, (tuple, list)):
            x = data[0]
        else:
            x = data

        batch_size = tf.shape(x)[0]

        # -- Random mask (TF-native, graph-mode compatible) ---------------
        mask_tf = _make_mask_tf(self._n_positions, self.mask_ratio, batch_size)

        with tf.GradientTape() as tape:
            # Context path: run full spectrogram through encoder,
            # then zero-out masked feature positions
            context_feat = self.context_encoder(x, training=True)   # (B, H, W, C)
            context_feat_masked = _apply_feature_mask(context_feat, mask_tf)

            # Flatten to sequence: (B, n_pos, C)
            ctx_flat = tf.reshape(context_feat_masked,
                                  (batch_size, self._n_positions, self._embed_dim))

            # Broadcast positional embeddings to batch
            pos_emb = tf.tile(
                tf.expand_dims(self.pos_embeddings, 0),   # (1, n_pos, pos_dim)
                [batch_size, 1, 1],                        # (B, n_pos, pos_dim)
            )

            # Predictor input: context + positional embedding
            pred_input = tf.concat([ctx_flat, pos_emb], axis=-1)  # (B, n_pos, C+P)
            pred_output = self.predictor(pred_input, training=True) # (B, n_pos, C)

            # Target path: full spectrogram through EMA encoder (no gradient)
            target_feat = self.target_encoder(x, training=False)   # (B, H, W, C)
            target_flat = tf.reshape(target_feat,
                                     (batch_size, self._n_positions, self._embed_dim))

            # Loss: MSE only at masked positions
            mask_f = tf.cast(mask_tf, tf.float32)[..., tf.newaxis]  # (B, n_pos, 1)
            diff = (pred_output - tf.stop_gradient(target_flat)) * mask_f
            loss = tf.reduce_sum(diff ** 2) / (
                tf.reduce_sum(mask_f) * tf.cast(self._embed_dim, tf.float32) + 1e-8
            )

        # Gradient update for context_encoder + predictor
        trainable_vars = (self.context_encoder.trainable_variables
                          + self.predictor.trainable_variables
                          + [self.pos_embeddings])
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # EMA update for target encoder
        self._ema_update()

        return {"loss": loss}

    def call(self, x, training=False):
        """Extract context encoder features (used for fine-tuning transfer)."""
        return self.context_encoder(x, training=training)


# ---------------------------------------------------------------------------
# Pretraining function
# ---------------------------------------------------------------------------

def pretrain_jepa(
    audio_files: list,
    cfg: BaseConfig,
    output_dir: str,
    epochs: int = 150,
    batch_size: int = 64,
    mask_ratio: float = 0.5,
    ema_decay: float = 0.996,
    learning_rate: float = 1e-3,
) -> JEPAPretrainer:
    """
    Self-supervised JEPA pretraining on unlabelled audio files.

    No labels needed — uses only raw .wav files to learn representations.

    :param audio_files: list of .wav file paths (can be from any class or unlabelled)
    :param cfg:         BaseConfig for DSP parameters (SAMPLE_RATE, N_MEL, etc.)
    :param output_dir:  directory to save pretrained encoder weights
    :param epochs:      pretraining epochs
    :param batch_size:  batch size
    :param mask_ratio:  fraction of feature positions to mask (0.5 = 50%)
    :param ema_decay:   EMA update decay for target encoder (0.996 recommended)
    :param learning_rate: optimizer learning rate
    :returns:           trained JEPAPretrainer instance
    """
    from ..inference.predict import MelExtractor

    os.makedirs(output_dir, exist_ok=True)

    # -- Feature extraction --------------------------------------------------
    extractor = MelExtractor(cfg)
    input_shape = extractor.input_shape
    print(f"\n  Audio-JEPA pretraining")
    print(f"  Input shape  : {input_shape}")
    print(f"  Audio files  : {len(audio_files)}")
    print(f"  Mask ratio   : {mask_ratio}")
    print(f"  EMA decay    : {ema_decay}")
    print(f"  Epochs       : {epochs}")

    print("\n  Extracting mel spectrograms...")
    windows = []
    for fp in audio_files:
        for w in extractor.extract_from_file(fp, augment=False):
            windows.append(w)
    if not windows:
        raise ValueError("No audio windows extracted — check audio_files paths.")

    X = np.array(windows)[..., np.newaxis].astype(np.float32)  # (N, H, W, 1)
    print(f"  Total windows : {len(X)}")

    # -- Build model ---------------------------------------------------------
    pretrainer = JEPAPretrainer(
        input_shape=input_shape,
        cfg=cfg,
        mask_ratio=mask_ratio,
        ema_decay=ema_decay,
    )
    pretrainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    print(f"  Feature map  : ({pretrainer._feat_H}, {pretrainer._feat_W}, "
          f"{pretrainer._embed_dim}) = {pretrainer._n_positions} positions")
    print(f"  Masked       : ~{int(pretrainer._n_positions * mask_ratio)} positions/step")

    # -- Training ------------------------------------------------------------
    dataset = tf.data.Dataset.from_tensor_slices(X) \
        .shuffle(len(X), seed=42) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    history = pretrainer.fit(dataset, epochs=epochs, verbose=1)

    # -- Save encoder weights ------------------------------------------------
    enc_path = os.path.join(output_dir, "jepa_encoder.weights.h5")
    pretrainer.context_encoder.save_weights(enc_path)
    print(f"\n  Context encoder weights saved: {enc_path}")

    # -- Save loss curve -----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("JEPA loss (MSE in embedding space)")
    ax.set_title("Audio-JEPA pretraining loss", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jepa_loss.png"), dpi=150)
    plt.close(fig)

    return pretrainer


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------

def fine_tune_from_jepa(
    encoder_weights_path: str,
    input_shape: tuple,
    n_classes: int,
    cfg: BaseConfig,
    freeze_epochs: int = 10,
) -> keras.Model:
    """
    Build a classification model and initialise its encoder from JEPA weights.

    Two-phase fine-tuning strategy:
        Phase 1 (freeze_epochs):  encoder frozen, only head trained.
        Phase 2:                  all layers unfrozen, low LR.

    Call run_pipeline() as usual — it will use this pre-warmed model.

    :param encoder_weights_path: path to jepa_encoder.weights.h5
    :param input_shape:          (N_MEL, n_frames, 1) — must match pretraining
    :param n_classes:            number of sound classes
    :param cfg:                  BaseConfig (must have same DSP params as pretraining)
    :param freeze_epochs:        epochs to train with encoder frozen (default 10)
    :returns:                    compiled Keras Model ready for training
    """
    # Build full classification model
    model = build_model(input_shape, n_classes, cfg)

    # Build the encoder body to extract matching layer weights
    encoder_body = build_encoder_body(input_shape, cfg)
    encoder_body.load_weights(encoder_weights_path)

    # Copy encoder weights into the full model by matching layer names
    enc_layer_names = {l.name for l in encoder_body.layers}
    transferred = 0
    for layer in model.layers:
        if layer.name in enc_layer_names:
            try:
                enc_layer = encoder_body.get_layer(layer.name)
                layer.set_weights(enc_layer.get_weights())
                transferred += 1
            except (ValueError, AttributeError):
                pass

    print(f"  JEPA weight transfer: {transferred} layers initialised "
          f"from {encoder_weights_path}")

    if freeze_epochs > 0:
        print(f"  Encoder frozen for first {freeze_epochs} epochs.")
        print("  Call model.trainable = True before phase-2 training.")
        # Freeze encoder layers (everything except Dense head + BN in head)
        for layer in model.layers:
            if layer.name in enc_layer_names:
                layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.LABEL_SMOOTHING),
            metrics=["accuracy"],
        )

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--data-dir", "-d", required=True,
              help="Directory of .wav files for pretraining (no labels needed).")
@click.option("--output-dir", "-o", default="./jepa_pretrained", show_default=True,
              help="Directory to save encoder weights and loss plot.")
@click.option("--config", "-c", default="mcu_match", show_default=True,
              type=click.Choice(["base", "resnet32", "mcu_match",
                                 "mel24_mcu", "mel28_mcu", "mel32_mcu"]),
              help="DSP config preset (must match the target inference config).")
@click.option("--epochs", default=150, show_default=True,
              help="Pretraining epochs.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--mask-ratio", default=0.5, show_default=True,
              help="Fraction of feature positions to mask (0.0–1.0).")
@click.option("--lr", default=1e-3, show_default=True,
              help="Optimizer learning rate.")
def pretrain_jepa_cmd(data_dir, output_dir, config, epochs, batch_size,
                      mask_ratio, lr):
    """
    Self-supervised Audio-JEPA pretraining on unlabelled .wav files.

    After pretraining, use fine_tune_from_jepa() to initialise a classifier
    with the learned encoder weights before supervised fine-tuning.

    Example:
        # Pretrain on all available audio (labelled or not):
        uv run pretrain-jepa --data-dir ./audio_files --config mcu_match

        # Fine-tune afterwards (in Python):
        from classification.models.jepa import fine_tune_from_jepa
        from classification.configs.mcu_match import MCUMatchConfig
        model = fine_tune_from_jepa(
            "jepa_pretrained/jepa_encoder.weights.h5", (20,32,1), 4,
            MCUMatchConfig())
    """
    from ..training.train import CONFIG_REGISTRY

    cfg = CONFIG_REGISTRY[config]()

    # Collect all .wav files recursively
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        raise click.ClickException(f"No .wav files found in {data_dir}")

    print(f"  Found {len(audio_files)} .wav files in {data_dir}")

    pretrain_jepa(
        audio_files=audio_files,
        cfg=cfg,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        learning_rate=lr,
    )
