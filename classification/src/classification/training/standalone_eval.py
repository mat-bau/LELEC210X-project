"""
Standalone test-set evaluation for a saved model.

Usage:
    uv run evaluate-model --model-dir ./data/models/my_run
    uv run evaluate-model --model-dir ./data/models/my_run --no-gradcam
    uv run evaluate-model --model-dir ./data/models/my_run --real-data /path/to/audio_files

Outputs (saved to {model_dir}/evaluation/):
    confusion_matrix.png
    per_class_metrics.png
    metrics.json
    gradcam_grid.png         (unless --no-gradcam)
    gradcam_misclassified.png (unless --no-gradcam)
"""
from __future__ import annotations

import json
import os
import pickle

import click
import matplotlib
matplotlib.use("Agg")   # headless-safe — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..inference.predict import MelExtractor, load_config
from ..data.dataset import _load_wav_dir
from ..training.evaluate import predict_with_tta
from ..models.gradcam import plot_gradcam_grid, plot_gradcam_misclassified


# ---------------------------------------------------------------------------
# Classname loading helper
# ---------------------------------------------------------------------------

def _load_classnames(model_dir: str, real_data_dir: str) -> list:
    """
    Load classnames from model_config.pkl (preferred) or scan REAL_DATA_DIR.

    Priority:
        1. model_config.pkl  (saved by train.py — most reliable)
        2. Scan REAL_DATA_DIR for subdirectories that have a 'test' subfolder
    """
    pkl_path = os.path.join(model_dir, "model_config.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        classnames = meta.get("classnames")
        if classnames:
            return list(classnames)

    # Fallback: scan REAL_DATA_DIR for class subdirs with a /test folder
    if real_data_dir and os.path.exists(real_data_dir):
        classnames = sorted(
            d for d in os.listdir(real_data_dir)
            if os.path.isdir(os.path.join(real_data_dir, d))
            and os.path.exists(os.path.join(real_data_dir, d, "test"))
        )
        if classnames:
            print(f"  Classnames inferred from REAL_DATA_DIR: {classnames}")
            return classnames

    raise FileNotFoundError(
        f"Cannot determine classnames.\n"
        f"  model_config.pkl not found in {model_dir}\n"
        f"  and no class subdirs with /test found in {real_data_dir}"
    )


# ---------------------------------------------------------------------------
# Plotting helpers (headless, saves to file)
# ---------------------------------------------------------------------------

def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            classnames: list, acc: float,
                            save_path: str) -> None:
    """Single confusion matrix saved to file (headless, no plt.show())."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classnames) * 1.6),
                                    max(5, len(classnames) * 1.4)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — Test set (TTA)\nAccuracy: {100*acc:.1f}%",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classnames))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classnames, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classnames, fontsize=10)
    thresh = cm.max() / 2
    for i in range(len(classnames)):
        for j in range(len(classnames)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _save_per_class_metrics(classnames: list, prec, rec, f1, sup,
                             save_path: str) -> None:
    """Per-class precision/recall/F1 bar chart saved to file."""
    x, w = np.arange(len(classnames)), 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w, prec, w, label="Precision", color="#2196F3")
    ax1.bar(x,     rec,  w, label="Recall",    color="#4CAF50")
    ax1.bar(x + w, f1,   w, label="F1",        color="#9C27B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classnames, rotation=20, ha="right")
    ax1.set_ylim(0, 1.18)
    ax1.set_title("Precision / Recall / F1 — Test set", fontweight="bold")
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_standalone_evaluation(
    model_dir: str,
    real_data_dir: str | None = None,
    save_dir: str | None = None,
    no_gradcam: bool = False,
    max_misclassified: int = 20,
) -> dict:
    """
    Evaluate a saved model on the test set and produce all visualisations.

    :param model_dir:         Directory containing best_model.keras + config.json
    :param real_data_dir:     Override cfg.REAL_DATA_DIR (optional)
    :param save_dir:          Output directory (default: {model_dir}/evaluation/)
    :param no_gradcam:        Skip GradCAM plots (fast mode)
    :param max_misclassified: Max misclassified samples to show in GradCAM
    :returns:                 metrics dict {class: {precision, recall, f1}, accuracy}
    """
    import keras

    # -- Setup ---------------------------------------------------------------
    save_dir = save_dir or os.path.join(model_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    SEP = "=" * 65
    print(f"\n{SEP}\n  STANDALONE EVALUATION\n  model_dir : {model_dir}\n{SEP}")

    # -- Config --------------------------------------------------------------
    cfg = load_config(model_dir)
    if real_data_dir:
        cfg.REAL_DATA_DIR = real_data_dir
    # Resolve REAL_DATA_DIR to an absolute path so that relative paths in
    # config.json (e.g. "../mcu/...") work regardless of the working directory.
    if not os.path.isabs(cfg.REAL_DATA_DIR):
        resolved = None
        base = model_dir
        for _ in range(8):  # remonte jusqu'à 8 niveaux
            candidate = os.path.normpath(os.path.join(base, cfg.REAL_DATA_DIR))
            if os.path.exists(candidate):
                resolved = candidate
                break
            base = os.path.dirname(base)
        if resolved is None:
            resolved = os.path.abspath(cfg.REAL_DATA_DIR)
            print(f"  WARNING: could not resolve REAL_DATA_DIR, using: {resolved}")
        cfg.REAL_DATA_DIR = resolved
    print(f"  REAL_DATA_DIR : {cfg.REAL_DATA_DIR}")
    print(f"  N_MEL={cfg.N_MEL}  n_frames={cfg.compute_n_frames()}")

    # -- Classnames ----------------------------------------------------------
    classnames = _load_classnames(model_dir, cfg.REAL_DATA_DIR)
    label_map = {c: i for i, c in enumerate(classnames)}
    n_classes = len(classnames)
    print(f"  Classes : {classnames}")

    # -- Model ---------------------------------------------------------------
    model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.keras not found in {model_dir}")
    print(f"  Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"  Parameters : {model.count_params():,}")

    # -- Feature extractor ---------------------------------------------------
    extractor = MelExtractor(cfg)
    print(f"  Input shape : {extractor.input_shape}")

    # -- Test set ------------------------------------------------------------
    print(f"\n  Loading test set from {cfg.REAL_DATA_DIR}/[class]/test/")
    sound_classes = [c for c in classnames if c != "background"]
    X_test, y_test = _load_wav_dir(
        cfg.REAL_DATA_DIR, sound_classes, label_map,
        extractor, aug_passes=0, subfolder="test",
    )
    X_test = X_test.astype(np.float32)
    print(f"  Test set : {len(X_test)} windows")

    if len(X_test) == 0:
        raise ValueError(
            f"No test samples found. Check that {cfg.REAL_DATA_DIR}/[class]/test/ "
            "directories exist and contain .wav files."
        )

    # -- Predictions (with TTA) ----------------------------------------------
    print(f"\n  Running inference (TTA x{cfg.TTA_STEPS})...")
    probs = predict_with_tta(model, X_test, cfg.TTA_STEPS)
    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y_test))
    print(f"  Test accuracy : {100*acc:.2f}%")

    # -- Per-class metrics ---------------------------------------------------
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred, labels=range(n_classes), average=None, zero_division=0)

    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}")
    print("  " + "-" * 56)
    for i, c in enumerate(classnames):
        print(f"  {c:<15} {prec[i]:>10.3f} {rec[i]:>10.3f} "
              f"{f1[i]:>8.3f} {int(sup[i]):>10}")
    print("  " + "-" * 56)
    print(f"  {'Macro avg':<15} {prec.mean():>10.3f} {rec.mean():>10.3f} "
          f"{f1.mean():>8.3f} {int(sup.sum()):>10}")

    # -- Save plots ----------------------------------------------------------
    print(f"\n  Saving plots to {save_dir}/")

    _save_confusion_matrix(
        y_test, y_pred, classnames, acc,
        os.path.join(save_dir, "confusion_matrix.png"),
    )

    _save_per_class_metrics(
        classnames, prec, rec, f1, sup,
        os.path.join(save_dir, "per_class_metrics.png"),
    )

    # -- Save metrics.json ---------------------------------------------------
    metrics = {
        "accuracy": acc,
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "n_test_samples": int(len(X_test)),
        "tta_steps": cfg.TTA_STEPS,
        "per_class": {
            c: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
            for i, c in enumerate(classnames)
        },
    }
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="ascii") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Saved: {metrics_path}")

    # -- GradCAM -------------------------------------------------------------
    if not no_gradcam:
        print("\n  Computing GradCAM visualisations...")
        try:
            plot_gradcam_grid(
                model, X_test, y_test, classnames, n_examples=3,
                save_path=os.path.join(save_dir, "gradcam_grid.png"),
            )
            plt.close("all")
            print(f"  Saved: {os.path.join(save_dir, 'gradcam_grid.png')}")

            # Cap misclassified to avoid huge figures
            wrong = np.where(y_test != y_pred)[0]
            if len(wrong) > max_misclassified:
                rng = np.random.default_rng(42)
                wrong = rng.choice(wrong, max_misclassified, replace=False)
                X_miss = X_test[wrong]
                y_miss_true = y_test[wrong]
                y_miss_pred = y_pred[wrong]
            else:
                X_miss, y_miss_true, y_miss_pred = X_test, y_test, y_pred

            plot_gradcam_misclassified(
                model, X_miss, y_miss_true, y_miss_pred, classnames,
                save_path=os.path.join(save_dir, "gradcam_misclassified.png"),
            )
            plt.close("all")
            print(f"  Saved: {os.path.join(save_dir, 'gradcam_misclassified.png')}")
        except Exception as exc:
            print(f"  WARNING GradCAM failed: {exc}")
    else:
        print("  GradCAM skipped (--no-gradcam).")

    print(f"\n{SEP}\n  DONE — results in {save_dir}\n{SEP}\n")
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model-dir", "-m", required=True,
              help="Directory with best_model.keras + config.json.")
@click.option("--real-data", "-r", default=None,
              help="Override REAL_DATA_DIR from the config.")
@click.option("--save-dir", "-s", default=None,
              help="Output directory (default: {model_dir}/evaluation/).")
@click.option("--no-gradcam", is_flag=True, default=False,
              help="Skip GradCAM plots for faster evaluation.")
@click.option("--max-misclassified", default=20, show_default=True,
              help="Max misclassified samples shown in GradCAM.")
def evaluate_model_cmd(model_dir, real_data, save_dir, no_gradcam,
                       max_misclassified):
    """Evaluate a saved ResNet model on the test set with all visualisations."""
    run_standalone_evaluation(
        model_dir=model_dir,
        real_data_dir=real_data,
        save_dir=save_dir,
        no_gradcam=no_gradcam,
        max_misclassified=max_misclassified,
    )
