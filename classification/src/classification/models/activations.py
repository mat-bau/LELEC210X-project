"""
Alternative activation functions for ResNet audio classifier.

Supported activations: relu, gelu, swish, mish, leaky_relu

Usage:
    from classification.models.activations import get_activation_fn
    act = get_activation_fn("swish")
    x = act(x)  # apply inside a Keras functional model
"""
from __future__ import annotations

import keras
from keras import layers
import tensorflow as tf


def _mish(x):
    """Mish: x * tanh(softplus(x))  — smoothest, best on small noisy datasets."""
    return x * tf.math.tanh(tf.math.softplus(x))


# Map of name -> Keras layer or Lambda
_ACTIVATION_MAP: dict = {
    "relu":       lambda: layers.ReLU(),
    "leaky_relu": lambda: layers.LeakyReLU(negative_slope=0.01),
    "gelu":       lambda: layers.Activation("gelu"),
    "swish":      lambda: layers.Activation("swish"),
    "mish":       lambda: layers.Lambda(_mish),
}

SUPPORTED_ACTIVATIONS = list(_ACTIVATION_MAP.keys())


def get_activation_fn(name: str) -> keras.layers.Layer:
    """
    Return a Keras layer instance for the requested activation.

    :param name: One of relu | gelu | swish | mish | leaky_relu
    :raises ValueError: if name is not supported
    """
    name = name.lower()
    if name not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {SUPPORTED_ACTIVATIONS}"
        )
    return _ACTIVATION_MAP[name]()
