"""
Model evaluation, history plots, and activation comparison.

Functions:
    plot_training_history(history, cfg)
    evaluate_model(model, X_train, y_train, X_val, y_val,
                   X_test, y_test, classnames, cfg) -> dict
    compare_activations(dataset, cfg_base, activations, epochs) -> dict
    analyze_false_positive_temporal(model, background_samples, cfg, extractor)
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import wandb
import keras
from keras.utils import to_categorical

from ..configs.base_config import BaseConfig


# ---------------------------------------------------------------------------
# Prediction with TTA
# ---------------------------------------------------------------------------

def predict_with_tta(model: keras.Model, X: np.ndarray, n_steps: int) -> np.ndarray:
    """Average predictions over n_steps slightly time-shifted copies of X."""
    probs = model.predict(X, verbose=0)
    for _ in range(n_steps - 1):
        X_aug = X.copy()
        shifts = np.random.randint(-3, 4, size=len(X))
        for i, s in enumerate(shifts):
            X_aug[i, :, :, 0] = np.roll(X_aug[i, :, :, 0], s, axis=1)
        probs += model.predict(X_aug, verbose=0)
    return probs / n_steps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_history(history, cfg: BaseConfig) -> None:
    """Plot accuracy and loss curves, mark best validation epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[metric], label="Train", lw=2)
        ax.plot(history.history[f"val_{metric}"], label="Val", lw=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    best = int(np.argmax(history.history["val_accuracy"]))
    axes[0].axvline(best, color="red", linestyle="--", alpha=0.5,
                    label=f"best epoch {best+1}")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.MODEL_DIR, "training_history.png"), dpi=150)
    plt.show()
    print(f"  Best val accuracy : "
          f"{100*history.history['val_accuracy'][best]:.2f}%  (epoch {best+1})")


def _plot_confusion_matrices(sets: list, classnames: list, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (title, y_true, y_pred, acc) in zip(axes, sets):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{title}\n{100*acc:.1f}%", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(classnames))
        ax.set_xticks(ticks)
        ax.set_xticklabels(classnames, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(classnames)
        thresh = cm.max() / 2
        for i in range(len(classnames)):
            for j in range(len(classnames)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Confusion matrices — Train / Val / Test", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def _plot_per_class_metrics(classnames, prec, rec, f1, sup) -> None:
    x, w = np.arange(len(classnames)), 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w, prec, w, label="Precision", color="#2196F3")
    ax1.bar(x,     rec,  w, label="Recall",    color="#4CAF50")
    ax1.bar(x + w, f1,   w, label="F1",        color="#9C27B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classnames, rotation=20, ha="right")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Precision / Recall / F1 per class", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        ax1.text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
        ax1.text(i,     r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax1.text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    bars = ax2.bar(classnames, sup, color="#FF9800", alpha=0.8)
    ax2.set_title("Support per class (test set)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, sup):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(int(s)), ha="center", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: keras.Model, X_train, y_train, X_val, y_val,
                   X_test, y_test, classnames: list, cfg: BaseConfig) -> dict:
    """Full evaluation with TTA, confusion matrices, per-class metrics, W&B logs."""
    n_classes = len(classnames)

    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_val_pred   = np.argmax(model.predict(X_val,   verbose=0), axis=1)
    y_test_probs = predict_with_tta(model, X_test, cfg.TTA_STEPS)
    y_test_pred  = np.argmax(y_test_probs, axis=1)

    acc_train = float(np.mean(y_train_pred == y_train))
    acc_val   = float(np.mean(y_val_pred   == y_val))
    acc_test  = float(np.mean(y_test_pred  == y_test))

    print(f"\n  Train : {100*acc_train:.2f}%")
    print(f"  Val   : {100*acc_val:.2f}%")
    print(f"  Test  : {100*acc_test:.2f}%  (TTA x{cfg.TTA_STEPS})")
    print(f"  Gap train/test : {100*(acc_train - acc_test):.2f}%")
    if   acc_train - acc_val > 0.15:
        print("  WARNING  significant overfitting (>15%)")
    elif acc_train - acc_val > 0.08:
        print("  WARNING  moderate overfitting (>8%)")
    else:
        print("  OK  good train/val balance")

    _plot_confusion_matrices([
        ("Training",   y_train, y_train_pred, acc_train),
        ("Validation", y_val,   y_val_pred,   acc_val),
        ("Test (TTA)", y_test,  y_test_pred,  acc_test),
    ], classnames, os.path.join(cfg.MODEL_DIR, "confusion_matrices.png"))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_test_pred, labels=range(n_classes), average=None)

    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}")
    print("-" * 58)
    for i, c in enumerate(classnames):
        print(f"  {c:<13} {prec[i]:>10.3f} {rec[i]:>10.3f} "
              f"{f1[i]:>8.3f} {int(sup[i]):>10}")
    print("-" * 58)
    print(f"  {'Macro avg':<13} {prec.mean():>10.3f} {rec.mean():>10.3f} "
          f"{f1.mean():>8.3f} {int(sup.sum()):>10}")

    _plot_per_class_metrics(classnames, prec, rec, f1, sup)

    for split_name, y_true, y_pred in [
        ("train", y_train, y_train_pred),
        ("val",   y_val,   y_val_pred),
        ("test",  y_test,  y_test_pred),
    ]:
        wandb.log({
            f"confusion/{split_name}": wandb.plot.confusion_matrix(
                y_true=y_true.tolist(), preds=y_pred.tolist(),
                class_names=classnames, title=f"Confusion - {split_name}",
            )
        })

    table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for i, c in enumerate(classnames):
        table.add_data(c, round(float(prec[i]), 3), round(float(rec[i]), 3),
                       round(float(f1[i]), 3), int(sup[i]))
    wandb.log({"metrics_per_class": table})

    return dict(
        train=acc_train, val=acc_val, test=acc_test,
        test_probs=y_test_probs, test_pred=y_test_pred,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Activation comparison (Task 4)
# ---------------------------------------------------------------------------

def compare_activations(dataset, cfg_base: BaseConfig, activations: list,
                        epochs: int = 50) -> dict:
    """
    Mini-run (epochs epochs) for each activation function, return val_accuracy dict.

    :param dataset: synthetic dataset (Feature_vector_DS compatible)
    :param cfg_base: base config — activation will be overridden per run
    :param activations: list of activation names to compare
    :param epochs: number of training epochs per activation
    :returns: {activation_name: best_val_accuracy}
    """
    from dataclasses import replace
    from ..inference.predict import MelExtractor
    from ..data.dataset import prepare_dataset
    from ..data.augmentation import MixupGenerator
    from ..models.resnet import build_model
    from ..training.callbacks import _build_callbacks
    from keras.utils import to_categorical

    results = {}
    for act_name in activations:
        print(f"\n{'='*60}\n  Activation: {act_name}\n{'='*60}")
        cfg = replace(cfg_base, ACTIVATION=act_name, EPOCHS=epochs,
                      MODEL_DIR=f"/tmp/act_cmp_{act_name}")
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)

        extractor = MelExtractor(cfg)
        X_train, X_val, _, y_train, y_val, _, classnames = \
            prepare_dataset(dataset, cfg, extractor)
        n_classes = len(classnames)

        model = build_model(extractor.input_shape, n_classes, cfg)
        train_gen = MixupGenerator(X_train, y_train, cfg.BATCH_SIZE,
                                   n_classes, alpha=cfg.MIXUP_ALPHA)
        y_val_cat = to_categorical(y_val, n_classes)
        lr_schedule, cbs = _build_callbacks(cfg, steps_per_epoch=len(train_gen))
        model.optimizer.learning_rate = lr_schedule

        history = model.fit(
            train_gen,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            callbacks=cbs,
            verbose=0,
        )
        best_val = float(max(history.history["val_accuracy"]))
        results[act_name] = best_val
        print(f"  {act_name:>12}  best val_acc = {100*best_val:.2f}%")

    # Summary plot
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    vals = [results[n] * 100 for n in names]
    bars = ax.bar(names, vals, color="#2196F3", alpha=0.85)
    ax.set_ylim(max(0, min(vals) - 5), 100)
    ax.set_ylabel("Best val accuracy (%)")
    ax.set_title(f"Activation comparison ({epochs} epochs)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()
    return results


# ---------------------------------------------------------------------------
# False positive temporal analysis (Task 3)
# ---------------------------------------------------------------------------

def analyze_false_positive_temporal(model: keras.Model,
                                    background_samples: np.ndarray,
                                    cfg: BaseConfig,
                                    extractor) -> None:
    """
    Simulate real-time blank-window behavior: run model on background samples,
    count packets that would exceed the confidence threshold and be wrongly
    submitted (false positives).

    Prints statistics and plots the distribution of max(P(sound class)) scores.
    """
    if len(background_samples) == 0:
        print("  No background samples to analyse.")
        return

    probs = model.predict(background_samples, verbose=0)  # (N, n_classes)
    classnames_minus_bg = None

    # Check if background is a class
    bg_col = None
    if hasattr(cfg, "BACKGROUND_DATA_DIR") and cfg.BACKGROUND_DATA_DIR:
        bg_col = 0  # background is always class 0

    if bg_col is not None:
        p_bg = probs[:, bg_col]
        p_sound_max = np.max(np.delete(probs, bg_col, axis=1), axis=1)
    else:
        p_bg = np.zeros(len(probs))
        p_sound_max = np.max(probs, axis=1)

    # Would-be submitted under RealTimeVoter thresholds
    would_submit = (p_bg < cfg.BACKGROUND_THRESHOLD) & \
                   (p_sound_max > cfg.CONFIDENCE_THRESHOLD)

    print(f"\n  False positive analysis on {len(background_samples)} background windows")
    print(f"  BACKGROUND_THRESHOLD  : {cfg.BACKGROUND_THRESHOLD}")
    print(f"  CONFIDENCE_THRESHOLD  : {cfg.CONFIDENCE_THRESHOLD}")
    print(f"  Would trigger submit  : {would_submit.sum()}  "
          f"({100*would_submit.mean():.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(p_sound_max, bins=30, color="#F44336", alpha=0.75, edgecolor="white")
    axes[0].axvline(cfg.CONFIDENCE_THRESHOLD, color="black", linestyle="--",
                    label=f"threshold={cfg.CONFIDENCE_THRESHOLD}")
    axes[0].set_xlabel("max P(sound class)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Background packets: max sound-class confidence",
                      fontweight="bold")
    axes[0].legend()

    if bg_col is not None:
        axes[1].hist(p_bg, bins=30, color="#2196F3", alpha=0.75, edgecolor="white")
        axes[1].axvline(cfg.BACKGROUND_THRESHOLD, color="black", linestyle="--",
                        label=f"threshold={cfg.BACKGROUND_THRESHOLD}")
        axes[1].set_xlabel("P(background)")
        axes[1].set_title("Background packets: P(background)", fontweight="bold")
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()
