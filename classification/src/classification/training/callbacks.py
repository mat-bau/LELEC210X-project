"""
Keras training callbacks.

GracefulStop  -- clean stop via flag file
_build_callbacks -- assembles full callback list with CosineDecayRestarts LR
"""
from __future__ import annotations

import os

import keras
from keras import callbacks
from wandb.integration.keras import WandbMetricsLogger

from ..configs.base_config import BaseConfig


class GracefulStop(keras.callbacks.Callback):
    """Stop training cleanly after the current epoch by writing a flag file."""

    FLAG = "./stop_training.flag"

    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(self.FLAG):
            print(f"\n  GracefulStop: flagged at epoch {epoch + 1}")
            os.remove(self.FLAG)
            self.model.stop_training = True


def _build_callbacks(cfg: BaseConfig, steps_per_epoch: int) -> tuple:
    """
    Build training callbacks and CosineDecayRestarts LR schedule.

    :returns: (lr_schedule, list_of_callbacks)
    """
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=cfg.LEARNING_RATE,
        first_decay_steps=steps_per_epoch * cfg.COSINE_RESTART_EPOCHS,
        t_mul=1.0,
        m_mul=cfg.COSINE_M_MUL,
        alpha=1e-6,
    )
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            os.path.join(cfg.MODEL_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        WandbMetricsLogger(log_freq="epoch"),
        GracefulStop(),
    ]
    return lr_schedule, cbs
