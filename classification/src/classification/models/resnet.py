"""
ResNet-small architecture for audio mel-spectrogram classification.

Functions:
    _residual_block(x, filters, stride, l2, activation_fn) -> tensor
    build_model(input_shape, n_classes, cfg) -> keras.Model
"""
from __future__ import annotations

import keras
from keras import layers, models, regularizers

from ..configs.base_config import BaseConfig
from .activations import get_activation_fn


def _residual_block(x, filters: int, stride: int = 1, l2: float = 1e-4,
                    activation_fn: keras.layers.Layer = None):
    """
    Single residual block: Conv2D -> BN -> Act -> Conv2D -> BN -> Add -> Act.

    :param activation_fn: if None, defaults to ReLU.
    """
    if activation_fn is None:
        activation_fn = layers.ReLU()

    reg = regularizers.l2(l2)
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_fn("relu") if activation_fn is None else \
        type(activation_fn).from_config(activation_fn.get_config())(x)

    # Use a fresh layer instance for each call (Keras functional API requirement)
    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 kernel_regularizer=reg, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    # Post-add activation — use a new instance of the same type
    cfg_copy = activation_fn.get_config() if hasattr(activation_fn, "get_config") \
        else {}
    x = activation_fn.__class__.from_config(cfg_copy)(x) \
        if hasattr(activation_fn.__class__, "from_config") \
        else layers.ReLU()(x)
    return x


def build_model(input_shape: tuple, n_classes: int, cfg: BaseConfig) -> keras.Model:
    """
    ResNet-small for audio spectrograms.

    Architecture:
        (N_MEL, T, 1) -> Stem(32,5x5) -> MaxPool ->
        ResBlock(64) x2 -> ResBlock(128, s=2) -> ResBlock(128) ->
        ResBlock(256, s=2) -> GAP -> Dense(256) -> BN -> Act ->
        Dropout -> Dense(n_classes, softmax)

    :param input_shape: (N_MEL, n_frames, 1)
    :param n_classes:   number of output classes
    :param cfg:         BaseConfig instance
    """
    act_name = cfg.ACTIVATION
    reg = regularizers.l2(cfg.L2_REG)
    inp = layers.Input(shape=input_shape)

    # -- Stem ----------------------------------------------------------------
    x = layers.Conv2D(32, 5, padding="same", kernel_regularizer=reg,
                      use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = get_activation_fn(act_name)(x)
    x = layers.MaxPool2D(2, padding="same")(x)

    # -- Residual blocks -----------------------------------------------------
    for filters, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2)]:
        x = _res_block(x, filters, stride=stride, l2=cfg.L2_REG,
                       act_name=act_name)

    # -- Head ----------------------------------------------------------------
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_fn(act_name)(x)
    x = layers.Dropout(cfg.DROPOUT_RATE)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="ResNet_Audio")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2),
        ],
    )
    n_params = model.count_params()
    print(f"  Parameters : {n_params:,}  (~{n_params * 4 / 1024:.0f} KB FP32)")
    return model


def _res_block(x, filters: int, stride: int = 1, l2: float = 1e-4,
               act_name: str = "relu"):
    """
    Residual block with configurable activation — internal helper.
    Uses fresh layer instances to satisfy Keras functional API.
    """
    reg = regularizers.l2(l2)
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_fn(act_name)(x)

    x = layers.Conv2D(filters, 3, padding="same",
                      kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 kernel_regularizer=reg, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = get_activation_fn(act_name)(x)
    return x
