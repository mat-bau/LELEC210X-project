"""
N_MEL sweep experiment: train one model per MCU-aligned N_MEL value and compare.

N_MEL candidates: 20 / 24 / 28 / 32  (all with MCU-aligned DSP params)
Each candidate is a feasible firmware configuration (MELVEC_LENGTH=N_MEL).

LimeSDR packet budget context:
    N_MEL=20 → 1280 bytes (current, upsampling=8 OK)
    N_MEL=24 → 1536 bytes
    N_MEL=28 → 1792 bytes
    N_MEL=32 → 2048 bytes  ← check RF quality before deploying

Usage:
    uv run sweep-mel --epochs 100 --output-dir ./mel_sweep_results
    uv run sweep-mel --epochs 5   --output-dir /tmp/sweep_smoke_test
    uv run sweep-mel --n-mel 20 24 32  # custom subset
"""
from __future__ import annotations

import json
import os
import time

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..configs.mcu_match import (
    MCUMatchConfig, Mel24MCUConfig, Mel28MCUConfig, Mel32MCUConfig,
)


# ---------------------------------------------------------------------------
# Config registry for the sweep
# ---------------------------------------------------------------------------

_SWEEP_CONFIGS = {
    20: MCUMatchConfig,
    24: Mel24MCUConfig,
    28: Mel28MCUConfig,
    32: Mel32MCUConfig,
}

# LimeSDR packet byte size for each N_MEL (N_MEL * 32_frames * 2_bytes)
_PACKET_BYTES = {n: n * 32 * 2 for n in _SWEEP_CONFIGS}


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _plot_sweep_results(results: list, output_dir: str) -> None:
    """
    Three-panel bar chart:
        1. Validation accuracy vs N_MEL
        2. Test accuracy vs N_MEL
        3. Lime packet size (bytes) vs N_MEL  (hardware budget indicator)
    """
    n_mels  = [r["n_mel"]    for r in results]
    val_acc = [r["val_acc"]  * 100 for r in results]
    tst_acc = [r["test_acc"] * 100 for r in results]
    pkt_b   = [_PACKET_BYTES.get(n, n * 32 * 2) for n in n_mels]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(results)]
    x = np.arange(len(results))
    w = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("N_MEL Sweep — MCU-aligned configs", fontsize=15, fontweight="bold")

    # Val accuracy
    bars = axes[0].bar(x, val_acc, w, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N_MEL={n}" for n in n_mels], rotation=20)
    axes[0].set_ylabel("Val accuracy (%)")
    axes[0].set_title("Validation accuracy")
    axes[0].set_ylim(max(0, min(val_acc) - 5), 100)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, val_acc):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{v:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")

    # Test accuracy
    bars = axes[1].bar(x, tst_acc, w, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"N_MEL={n}" for n in n_mels], rotation=20)
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Test accuracy (TTA)")
    axes[1].set_ylim(max(0, min(tst_acc) - 5), 100)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, tst_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{v:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")

    # Lime packet size
    bars = axes[2].bar(x, pkt_b, w, color="#F44336", alpha=0.7, edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"N_MEL={n}" for n in n_mels], rotation=20)
    axes[2].set_ylabel("Packet payload (bytes)")
    axes[2].set_title("LimeSDR packet size\n(check upsampling budget)")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, pkt_b):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, f"{v}B",
                     ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "mel_sweep_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------

def run_mel_sweep(
    dataset,
    n_mel_list: list,
    epochs: int,
    output_dir: str,
    real_data: str | None = None,
    author: str = "sweep",
) -> list:
    """
    Train one model per N_MEL value (MCU-aligned DSP params) and compare.

    :param dataset:    synthetic dataset passed to run_pipeline()
    :param n_mel_list: list of N_MEL values to test (subset of {20, 24, 28, 32})
    :param epochs:     training epochs per model (override config default)
    :param output_dir: root directory for all sweep outputs
    :param real_data:  override REAL_DATA_DIR (optional)
    :param author:     prefix for W&B run names
    :returns:          list of result dicts [{n_mel, val_acc, test_acc, model_dir}]
    """
    from ..training.train import run_pipeline

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for n_mel in n_mel_list:
        if n_mel not in _SWEEP_CONFIGS:
            raise ValueError(
                f"N_MEL={n_mel} not in sweep registry. "
                f"Available: {sorted(_SWEEP_CONFIGS.keys())}"
            )

        print(f"\n{'='*65}")
        print(f"  SWEEP  N_MEL={n_mel}")
        print(f"{'='*65}")

        cfg = _SWEEP_CONFIGS[n_mel]()
        cfg.EPOCHS = epochs
        cfg.MODEL_DIR = os.path.join(output_dir, f"sweep_mel{n_mel}")
        if real_data:
            cfg.REAL_DATA_DIR = real_data

        date_str = time.strftime("%m%d_%H%M")
        run_name = f"{author}_mel{n_mel}_{date_str}"

        _, metrics, _, _ = run_pipeline(dataset, cfg, run_name=run_name,
                                        tags=[f"sweep", f"N_MEL={n_mel}"])

        result = {
            "n_mel":     n_mel,
            "val_acc":   metrics["val"],
            "test_acc":  metrics["test"],
            "model_dir": cfg.MODEL_DIR,
            "n_frames":  cfg.compute_n_frames(),
            "packet_bytes": _PACKET_BYTES.get(n_mel, n_mel * 32 * 2),
        }
        results.append(result)
        print(f"\n  N_MEL={n_mel}  val={100*metrics['val']:.2f}%  "
              f"test={100*metrics['test']:.2f}%")

    # -- Save results --------------------------------------------------------
    results_path = os.path.join(output_dir, "sweep_results.json")
    with open(results_path, "w", encoding="ascii") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved: {results_path}")

    # -- Comparison plot -----------------------------------------------------
    if len(results) > 1:
        _plot_sweep_results(results, output_dir)

    # -- Summary table -------------------------------------------------------
    print(f"\n  {'N_MEL':>8}  {'Val %':>8}  {'Test %':>8}  {'Packet':>10}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {r['n_mel']:>8}  {100*r['val_acc']:>8.2f}  "
              f"{100*r['test_acc']:>8.2f}  {r['packet_bytes']:>8}B")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--epochs", default=100, show_default=True,
              help="Training epochs per model.")
@click.option("--output-dir", default="./mel_sweep_results", show_default=True,
              help="Root output directory for sweep results.")
@click.option("--real-data", default=None,
              help="Override REAL_DATA_DIR.")
@click.option("--author", default="sweep", show_default=True,
              help="Author prefix for W&B run names.")
@click.option("--n-mel", multiple=True, type=int, default=(20, 24, 28, 32),
              show_default=True,
              help="N_MEL values to sweep. Can be repeated: --n-mel 20 --n-mel 32")
def sweep_mel_cmd(epochs, output_dir, real_data, author, n_mel):
    """
    Train one MCU-aligned model per N_MEL and compare accuracy vs packet size.

    Each N_MEL value corresponds to a different firmware configuration
    (MELVEC_LENGTH). All configs use SAMPLE_RATE=10200, HOP_LENGTH=512,
    N_FFT=512 to match the MCU DSP pipeline exactly.
    """
    from classification.datasets import Dataset

    dataset = Dataset()
    for cls in ("background", "birds", "handsaw", "helicopter"):
        try:
            dataset.remove_class(cls)
        except KeyError:
            pass

    run_mel_sweep(
        dataset=dataset,
        n_mel_list=list(n_mel),
        epochs=epochs,
        output_dir=output_dir,
        real_data=real_data,
        author=author,
    )
