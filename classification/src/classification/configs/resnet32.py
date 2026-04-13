"""
ResNet32 config: N_MEL=32, suitable for experiments with smaller mel resolution.

Usage:
    from classification.configs.resnet32 import ResNet32Config
    cfg = ResNet32Config(MODEL_DIR="./data/models/resnet32/run1")
"""
from __future__ import annotations

from dataclasses import dataclass

from .base_config import BaseConfig


@dataclass
class ResNet32Config(BaseConfig):
    # Smaller mel resolution — faster training, tests domain-gap sensitivity
    N_MEL: int = 32
    DROPOUT_RATE: float = 0.3
    MIXUP_ALPHA: float = 0.4
    COSINE_RESTART_EPOCHS: int = 40
    # Background class support — set BACKGROUND_DATA_DIR to activate
    BACKGROUND_DATA_DIR: str = ""
    # Slightly higher threshold for 32-mel (less discriminative features)
    CONFIDENCE_THRESHOLD: float = 0.6
    BACKGROUND_THRESHOLD: float = 0.4
