"""
GradCAM visualisation for audio mel-spectrogram classifiers.

Functions:
    auto_detect_last_conv_layer(model)            -> str
    compute_gradcam(model, spec, class_idx, layer) -> np.ndarray heatmap [0,1]
    plot_gradcam_grid(model, X, y_true, classnames, n_examples, save_path)
    plot_gradcam_misclassified(model, X, y_true, y_pred, classnames, save_path)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras


# ---------------------------------------------------------------------------
# Layer detection
# ---------------------------------------------------------------------------

def auto_detect_last_conv_layer(model: keras.Model) -> str:
    """
    Find the last convolutional ReLU-type layer before GlobalAveragePooling2D.

    Heuristic: iterate layers in reverse, return the first layer whose name
    contains 're_lu', 'relu', 'swish', 'gelu', 'mish', or 'leaky' AND whose
    output rank is 4 (batch, H, W, C), stopping before GAP.
    """
    activation_keywords = ("re_lu", "relu", "swish", "gelu", "mish", "leaky",
                            "activation", "lambda")
    gap_found = False
    for layer in reversed(model.layers):
        lname = layer.name.lower()
        if "global_average" in lname:
            gap_found = True
            continue
        if not gap_found:
            continue
        try:
            out_shape = layer.output_shape
        except AttributeError:
            continue
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if len(out_shape) != 4:
            continue
        if any(kw in lname for kw in activation_keywords):
            return layer.name
    # Fallback: last Conv2D before GAP
    for layer in reversed(model.layers):
        lname = layer.name.lower()
        if "global_average" in lname:
            gap_found = True
            continue
        if not gap_found:
            continue
        if "conv2d" in lname:
            return layer.name
    raise RuntimeError("Could not detect a suitable last conv layer in the model.")


# ---------------------------------------------------------------------------
# GradCAM core
# ---------------------------------------------------------------------------

def compute_gradcam(model: keras.Model, spectrogram: np.ndarray,
                    class_idx: int,
                    last_conv_layer_name: str) -> np.ndarray:
    """
    Compute GradCAM heatmap for one spectrogram.

    :param spectrogram: shape (N_MEL, n_frames, 1) — single sample, no batch dim
    :param class_idx:   index of the target class
    :param last_conv_layer_name: name of the conv layer to visualise
    :return: heatmap of shape (N_MEL, n_frames) normalised to [0, 1]
    """
    # Build grad model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    inp = tf.expand_dims(
        tf.cast(spectrogram, tf.float32), axis=0
    )  # (1, H, W, 1)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inp, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)           # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                       # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)                        # (h, w)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to input resolution via bilinear interpolation
    h_in, w_in = spectrogram.shape[:2]
    heatmap_resized = tf.image.resize(
        heatmap[np.newaxis, ..., np.newaxis],
        [h_in, w_in],
        method="bilinear",
    )[0, :, :, 0].numpy()

    return heatmap_resized.astype(np.float32)


def _overlay(spec: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap (jet colormap) on top of spectrogram (grayscale)."""
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec_rgb = np.stack([spec_norm] * 3, axis=-1)          # (H, W, 3)
    jet = cm.get_cmap("jet")
    heat_rgb = jet(heatmap)[..., :3]                       # (H, W, 3)
    overlaid = (1 - alpha) * spec_rgb + alpha * heat_rgb
    return np.clip(overlaid, 0, 1)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_gradcam_grid(model: keras.Model, X: np.ndarray, y_true: np.ndarray,
                      classnames: list, n_examples: int = 3,
                      save_path: str = None) -> None:
    """
    Grid of n_classes x n_examples x 2 (spectrogram | GradCAM overlay).

    Rows = classes, columns = examples x 2 (raw spec + overlay).
    """
    last_conv = auto_detect_last_conv_layer(model)
    n_classes = len(classnames)
    n_cols = n_examples * 2
    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(n_cols * 2.5, n_classes * 2.5))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for row, cls_idx in enumerate(range(n_classes)):
        idxs = np.where(y_true == cls_idx)[0][:n_examples]
        for col_pair, sample_idx in enumerate(idxs):
            spec = X[sample_idx, :, :, 0]
            hmap = compute_gradcam(model, X[sample_idx], cls_idx, last_conv)
            overlay = _overlay(spec, hmap)

            ax_spec = axes[row, col_pair * 2]
            ax_over = axes[row, col_pair * 2 + 1]

            ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="magma")
            ax_over.imshow(overlay, aspect="auto", origin="lower")

            if col_pair == 0:
                ax_spec.set_ylabel(classnames[cls_idx], fontsize=9, fontweight="bold")
            for ax in (ax_spec, ax_over):
                ax.set_xticks([])
                ax.set_yticks([])

        # Fill unused cells
        for col_pair in range(len(idxs), n_examples):
            axes[row, col_pair * 2].axis("off")
            axes[row, col_pair * 2 + 1].axis("off")

    plt.suptitle("GradCAM — spectrogram (left) | heatmap overlay (right)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gradcam_misclassified(model: keras.Model, X: np.ndarray,
                               y_true: np.ndarray, y_pred: np.ndarray,
                               classnames: list,
                               save_path: str = None) -> None:
    """
    GradCAM for misclassified samples.

    Shows spec + heatmap for true class and predicted class side-by-side.
    Focuses attention on fire/fireworks confusion and similar boundary errors.
    """
    last_conv = auto_detect_last_conv_layer(model)
    wrong_idxs = np.where(y_true != y_pred)[0]

    if len(wrong_idxs) == 0:
        print("  No misclassified samples found.")
        return

    n_cols = 4  # raw | true heatmap | raw | pred heatmap
    fig, axes = plt.subplots(len(wrong_idxs), n_cols,
                             figsize=(n_cols * 2.5, len(wrong_idxs) * 2.5))
    if len(wrong_idxs) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(wrong_idxs):
        spec = X[idx, :, :, 0]
        true_idx = int(y_true[idx])
        pred_idx = int(y_pred[idx])

        hmap_true = compute_gradcam(model, X[idx], true_idx, last_conv)
        hmap_pred = compute_gradcam(model, X[idx], pred_idx, last_conv)

        axes[row, 0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[row, 0].set_title(f"True: {classnames[true_idx]}", fontsize=8)
        axes[row, 1].imshow(_overlay(spec, hmap_true), aspect="auto", origin="lower")
        axes[row, 1].set_title("GradCAM (true class)", fontsize=8)

        axes[row, 2].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[row, 2].set_title(f"Pred: {classnames[pred_idx]}", fontsize=8, color="red")
        axes[row, 3].imshow(_overlay(spec, hmap_pred), aspect="auto", origin="lower")
        axes[row, 3].set_title("GradCAM (pred class)", fontsize=8, color="red")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("GradCAM — misclassified samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
