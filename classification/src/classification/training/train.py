"""
Training pipeline entry point.

Can be used as a module:
    python3 -m classification.training.train \\
        --config resnet32 \\
        --author mateo \\
        --tags "background-class,32mel,swish"

Or imported programmatically:
    from classification.training.train import run_pipeline
"""
from __future__ import annotations

import os
import pickle
import time

import click
import keras
import numpy as np
import wandb
from keras.utils import to_categorical

from ..configs.base_config import BaseConfig
from ..configs.resnet32 import ResNet32Config
from ..data.augmentation import MixupGenerator
from ..data.dataset import prepare_dataset
from ..inference.predict import MelExtractor
from ..models.resnet import build_model
from ..training.callbacks import _build_callbacks
from ..training.evaluate import evaluate_model, plot_training_history


# ---------------------------------------------------------------------------
# Config registry — add new configs here
# ---------------------------------------------------------------------------

CONFIG_REGISTRY: dict = {
    "base":     BaseConfig,
    "resnet32": ResNet32Config,
}


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------

def _init_wandb(cfg: BaseConfig, run_name: str, tags: list):
    return wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name=run_name,
        tags=tags,
        settings=wandb.Settings(init_timeout=300),
        config=cfg.wandb_config(),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: keras.Model, X_train, y_train, X_val, y_val,
                cfg: BaseConfig, n_classes: int):
    """Full training loop with MixupGenerator and CosineDecayRestarts."""
    train_gen = MixupGenerator(X_train, y_train, cfg.BATCH_SIZE,
                               n_classes, alpha=cfg.MIXUP_ALPHA)
    y_val_cat = to_categorical(y_val, n_classes)

    lr_schedule, cbs = _build_callbacks(cfg, steps_per_epoch=len(train_gen))
    model.optimizer.learning_rate = lr_schedule

    print(f"\n  Train : {len(X_train)} samples | {len(train_gen)} batches/epoch")
    print(f"  Val   : {len(X_val)} samples")

    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val_cat),
        epochs=cfg.EPOCHS,
        callbacks=cbs,
        verbose=1,
    )
    with open(os.path.join(cfg.MODEL_DIR, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    return history


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model_and_metadata(model: keras.Model, classnames: list,
                             extractor: MelExtractor, metrics: dict,
                             cfg: BaseConfig) -> None:
    """
    Save model + metadata.

    Creates:
        best_model.keras
        best_model.weights.h5
        model_config.pkl   (backward compat)
        config.json        (human-readable, preferred)
    """
    import json as _json
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    model_path = os.path.join(cfg.MODEL_DIR, "best_model.keras")
    model.save(model_path)
    model.save_weights(os.path.join(cfg.MODEL_DIR, "best_model.weights.h5"))

    # config.json (preferred)
    cfg.to_json(os.path.join(cfg.MODEL_DIR, "config.json"))

    # model_config.pkl (backward compat for existing uart-reader.py)
    meta = dict(
        classnames=classnames,
        input_shape=extractor.input_shape,
        n_mel=cfg.N_MEL,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        sample_rate=cfg.SAMPLE_RATE,
        test_accuracy=metrics["test"],
        keras_version=keras.__version__,
    )
    with open(os.path.join(cfg.MODEL_DIR, "model_config.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Supplementary metrics.json
    metrics_clean = {k: v for k, v in metrics.items()
                     if not isinstance(v, np.ndarray)}
    with open(os.path.join(cfg.MODEL_DIR, "metrics.json"), "w",
              encoding="ascii") as f:
        _json.dump(metrics_clean, f, indent=2)

    print(f"  Model saved to {model_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(synth_dataset, cfg: BaseConfig,
                 run_name: str = None, tags: list = None) -> tuple:
    """
    Orchestrate the full training pipeline.

    :returns: (best_model, metrics, classnames, extractor)
    """
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    run_name = run_name or f"resnet_{time.strftime('%m%d_%H%M')}"
    run = _init_wandb(cfg, run_name, tags or [])

    extractor = MelExtractor(cfg)
    print(f"\n  Input shape : {extractor.input_shape}")

    X_train, X_val, X_test, y_train, y_val, y_test, classnames = \
        prepare_dataset(synth_dataset, cfg, extractor)

    n_classes = len(classnames)
    model = build_model(extractor.input_shape, n_classes, cfg)
    wandb.log({"n_params": model.count_params()})

    history = train_model(model, X_train, y_train, X_val, y_val, cfg, n_classes)
    plot_training_history(history, cfg)

    best_model = keras.models.load_model(
        os.path.join(cfg.MODEL_DIR, "best_model.keras"))

    metrics = evaluate_model(
        best_model, X_train, y_train, X_val, y_val,
        X_test, y_test, classnames, cfg,
    )

    wandb.log({
        "final/train_acc": metrics["train"],
        "final/val_acc":   metrics["val"],
        "final/test_acc":  metrics["test"],
    })

    # Log config.json + model as W&B artifacts
    artifact_cfg = wandb.Artifact("config", type="config")
    artifact_cfg.add_file(os.path.join(cfg.MODEL_DIR, "config.json"))
    run.log_artifact(artifact_cfg)

    artifact_model = wandb.Artifact("best_model", type="model")
    artifact_model.add_file(os.path.join(cfg.MODEL_DIR, "best_model.keras"))
    run.log_artifact(artifact_model)
    wandb.finish()

    save_model_and_metadata(best_model, classnames, extractor, metrics, cfg)

    print(f"\n  Test accuracy (TTA x{cfg.TTA_STEPS}) : {100*metrics['test']:.2f}%")
    return best_model, metrics, classnames, extractor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "-c", default="base",
              type=click.Choice(list(CONFIG_REGISTRY.keys())),
              show_default=True,
              help="Config preset to use.")
@click.option("--author", "-a", default="anon", show_default=True,
              help="Author name (used in model dir naming).")
@click.option("--tags", "-t", default="",
              help="Comma-separated W&B tags.")
@click.option("--model-dir", "-d", default=None,
              help="Override MODEL_DIR. Default: auto-generated from author/config/date.")
@click.option("--real-data", default=None,
              help="Override REAL_DATA_DIR.")
def main(config: str, author: str, tags: str, model_dir: str,
         real_data: str) -> None:
    """Train the ResNet audio classifier."""
    from classification.datasets import Dataset

    cfg = CONFIG_REGISTRY[config]()

    date_str = time.strftime("%Y%m%d_%H%M")
    if model_dir:
        cfg.MODEL_DIR = model_dir
    else:
        cfg.MODEL_DIR = f"./data/models/{author}_{config}_{date_str}"

    if real_data:
        cfg.REAL_DATA_DIR = real_data

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    run_name = f"{author}_{config}_{date_str}"

    dataset = Dataset()
    for cls in ("background", "birds", "handsaw", "helicopter"):
        try:
            dataset.remove_class(cls)
        except KeyError:
            pass

    run_pipeline(dataset, cfg, run_name=run_name, tags=tag_list)


if __name__ == "__main__":
    main()
