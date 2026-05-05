"""
Big hyperparameter sweep: N_MEL × n_frames × dropout × mixup_alpha × aug_profile.

432 combinations total (4 × 4 × 3 × 3 × 3).
Saves results incrementally after each run (crash-safe).
Re-running the same --output-dir skips already-completed combos.

Usage:
    cd classification
    uv run big-sweep --epochs 150 --patience 30 --output-dir ./big_sweep_results

    # Smoke test (very fast)
    uv run big-sweep --epochs 5 --patience 5 --output-dir /tmp/sweep_smoke

    # Resume after crash (completed combos are skipped automatically)
    uv run big-sweep --output-dir ./big_sweep_results

    # Restrict to a sub-grid (repeatable flags)
    uv run big-sweep --n-mel 24 --aug-profile standard
"""
from __future__ import annotations

import json
import os
import time
from itertools import product

import click

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt  # noqa: E402  (after backend set)

from ..configs.mcu_match import MCUMatchConfig
from ..training.train import run_pipeline


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

SWEEP_GRID: dict = {
    "n_mel":       [20, 24, 28, 32],
    "n_melvecs":   [20, 26, 32, 38],
    "dropout":     [0.2],
    "mixup_alpha": [0.2],
    "aug_profile": ["light", "standard"],
}

# DURATION_MS such that compute_n_frames() == n_melvecs.
# formula (MCUMatchConfig): int(DURATION_MS/1000 * 10200/512) + 1 == n_melvecs
# SAMPLE_RATE=10200, HOP_LENGTH=512 (firmware-aligned, no overlap)
_DURATION_FOR_FRAMES: dict[int, int] = {
    20: 1000,   # int(1.000 * 19.921) + 1 = 19 + 1 = 20 ✓
    26: 1255,   # int(1.255 * 19.921) + 1 = 25 + 1 = 26 ✓
    32: 1557,   # int(1.557 * 19.921) + 1 = 31 + 1 = 32 ✓
    38: 1858,   # int(1.858 * 19.921) + 1 = 37 + 1 = 38 ✓
}

_AUG_PROFILES: dict[str, dict] = {
    # TIME_SHIFT + NOISE only (fast, no pitch/stretch)
    "light": dict(
        ENABLE_TIME_SHIFT=True,
        ENABLE_NOISE=True,
        ENABLE_SPEC_MASKING=False,
        ENABLE_PITCH_SHIFT=False,
        ENABLE_TIME_STRETCH=False,
    ),
    # Adds SpecAugment masking (current MCUMatchConfig default)
    "standard": dict(
        ENABLE_TIME_SHIFT=True,
        ENABLE_NOISE=True,
        ENABLE_SPEC_MASKING=True,
        ENABLE_PITCH_SHIFT=False,
        ENABLE_TIME_STRETCH=False,
    ),
    # All augmentations enabled (slowest per epoch)
    "full": dict(
        ENABLE_TIME_SHIFT=True,
        ENABLE_NOISE=True,
        ENABLE_SPEC_MASKING=True,
        ENABLE_PITCH_SHIFT=True,
        ENABLE_TIME_STRETCH=True,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tag(n_mel: int, n_melvecs: int, dropout: float,
              mixup: float, aug: str, norm_mode: str = "db",
              bg_overlay: bool = False) -> str:
    """Unique short identifier for a combo — used as folder name and W&B tag."""
    tag = (f"mel{n_mel}_fr{n_melvecs}"
           f"_do{int(dropout * 100)}"
           f"_mx{int(mixup * 100)}"
           f"_{aug}"
           f"_{norm_mode}")
    if bg_overlay:
        tag += "_bgov"
    return tag


def _total_combos(grid: dict) -> int:
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _plot_results(results: list[dict], output_dir: str) -> None:
    """
    Generate summary visualisations from sweep results.

    Produces:
        big_sweep_heatmaps.png  — 4 seaborn pivot-table heatmaps
        top10_bar.png           — horizontal bar chart of top-10 configs
        best_config.json        — top-10 configs with packet_bytes for filtering
    """
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        print("  pandas / seaborn not available — skipping plots")
        return

    df = pd.DataFrame(results)
    df = df.dropna(subset=["test_acc"])
    if df.empty:
        print("  No successful runs yet — skipping plots.")
        return

    # ---- 4 heatmaps ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Big Sweep — mean test accuracy ({len(df)} runs)",
        fontsize=14, fontweight="bold",
    )

    def _heatmap(ax, row_col: str, col_col: str, title: str) -> None:
        pivot = df.groupby([row_col, col_col])["test_acc"].mean().unstack()
        import seaborn as sns
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="YlGn", vmin=0.55, vmax=1.0,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)

    _heatmap(axes[0, 0], "n_mel",     "n_melvecs",  "N_MEL × n_frames")
    _heatmap(axes[0, 1], "dropout",   "mixup_alpha", "dropout × mixup_alpha")
    _heatmap(axes[1, 0], "n_mel",     "aug_profile", "N_MEL × aug_profile")
    _heatmap(axes[1, 1], "n_melvecs", "aug_profile", "n_frames × aug_profile")

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "big_sweep_heatmaps.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmaps      → {heatmap_path}")

    # ---- Top-10 bar chart ----
    top10 = df.nlargest(10, "test_acc").reset_index(drop=True)
    top10["label"] = top10.apply(
        lambda r: _make_tag(
            int(r.n_mel), int(r.n_melvecs),
            r.dropout, r.mixup_alpha, r.aug_profile,
        ),
        axis=1,
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.barh(top10["label"][::-1], top10["test_acc"][::-1],
                    color="steelblue", edgecolor="white")
    ax2.set_xlabel("Test accuracy")
    ax2.set_title(f"Top-10 configs (of {len(df)} runs)")
    ax2.set_xlim(max(0.5, top10["test_acc"].min() - 0.05), 1.0)
    for i, row in top10[::-1].reset_index(drop=True).iterrows():
        ax2.text(
            row["test_acc"] + 0.002, i,
            f"{row['test_acc']:.3f}  ({int(row['packet_bytes'])}B)",
            va="center", fontsize=8,
        )
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "top10_bar.png")
    fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Top-10 chart  → {bar_path}")

    # ---- best_config.json ----
    top10_export = top10.drop(columns=["label"], errors="ignore")
    best_path = os.path.join(output_dir, "best_config.json")
    top10_export.to_json(best_path, orient="records", indent=2)
    print(f"  Best configs  → {best_path}")

    print("\n  Top 3:")
    for _, row in top10.head(3).iterrows():
        print(f"    {row['label']:45s}  "
              f"test={row['test_acc']:.4f}  "
              f"val={row['val_acc']:.4f}  "
              f"{int(row['packet_bytes'])}B")


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_big_sweep(
    dataset,
    grid: dict,
    epochs: int,
    patience: int,
    output_dir: str,
    real_data: str | None,
    author: str,
    norm_mode: str = "db",
    bg_overlay: bool = False,
    bg_dir: str = "",
) -> list[dict]:
    """
    Iterate over all combinations in *grid*, train one model per combo,
    and save sweep_results.json after every run.

    Already-completed combos (identified by their *tag*) are skipped
    automatically, enabling crash-safe resumption.

    :returns: full list of result dicts (including previously saved runs)
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "sweep_results.json")

    # -- crash-safe resume ---------------------------------------------------
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results: list[dict] = json.load(f)
        done_tags = {r["tag"] for r in results if "tag" in r}
        print(f"  Resuming sweep — {len(results)} runs already completed.")
    else:
        results = []
        done_tags: set[str] = set()

    combos = list(product(*grid.values()))
    keys   = list(grid.keys())
    total  = len(combos)

    print(f"\n  Big sweep: {total} combinations (grid filtered)")
    print(f"  EPOCHS={epochs}, EARLY_STOPPING_PATIENCE={patience}")
    print(f"  USE_SE_BLOCKS=True (fixed), SAMPLE_RATE=10200, HOP_LENGTH=512")
    _norm_desc = {
        "db":         "fixed dB/80 → [-1, 0]",
        "l2":         "L2 norm (amplitude-invariant)",
        "zscore":     "per-band z-score (WARNING: amplifies silent-band noise)",
        "mcu_linear": "linear magnitude, Hamming, /max → [0, 1]  ← matches MCU DSP",
    }
    print(f"  NORM_MODE={norm_mode!r}  ({_norm_desc.get(norm_mode, norm_mode)})")
    if bg_overlay:
        print(f"  BG_OVERLAY=True  dir={bg_dir or '(from cfg.BACKGROUND_DATA_DIR)'}")
    print(f"  Output dir: {output_dir}\n")

    for i, values in enumerate(combos):
        params    = dict(zip(keys, values))
        n_mel     = params["n_mel"]
        n_melvecs = params["n_melvecs"]
        dropout   = params["dropout"]
        mixup     = params["mixup_alpha"]
        aug       = params["aug_profile"]

        tag = _make_tag(n_mel, n_melvecs, dropout, mixup, aug, norm_mode, bg_overlay)

        if tag in done_tags:
            print(f"[{i+1}/{total}] SKIP: {tag}")
            continue

        print(f"\n[{i+1}/{total}/{total}] START: {tag}")
        print(f"  n_mel={n_mel}  n_frames={n_melvecs}  dropout={dropout}"
              f"  mixup={mixup}  aug={aug}"
              f"  packet={2*n_mel*n_melvecs}B")

        # -- build config ----------------------------------------------------
        cfg = MCUMatchConfig()
        cfg.N_MEL                 = n_mel
        cfg.DURATION_MS           = _DURATION_FOR_FRAMES[n_melvecs]
        cfg.DROPOUT_RATE          = dropout
        cfg.MIXUP_ALPHA           = mixup
        cfg.EPOCHS                = epochs
        cfg.EARLY_STOPPING_PATIENCE = patience
        cfg.USE_SE_BLOCKS         = True
        cfg.NORM_MODE             = norm_mode
        cfg.ENABLE_BG_OVERLAY     = bg_overlay
        if bg_dir:
            cfg.BACKGROUND_DATA_DIR = bg_dir
        cfg.MODEL_DIR             = os.path.join(output_dir, tag)

        for attr, val in _AUG_PROFILES[aug].items():
            setattr(cfg, attr, val)

        if real_data:
            cfg.REAL_DATA_DIR = real_data

        run_name = f"{author}_{tag}_{time.strftime('%m%d_%H%M')}"

        # -- train -----------------------------------------------------------
        try:
            _, metrics, _, _ = run_pipeline(
                dataset, cfg,
                run_name=run_name,
                tags=["big-sweep", f"aug-{aug}", tag,
                      f"mel{n_mel}", f"fr{n_melvecs}"],
            )
            val_acc  = float(metrics.get("val",  0.0) or 0.0)
            test_acc = float(metrics.get("test", 0.0) or 0.0)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            val_acc = test_acc = None

        # -- record & incremental save ---------------------------------------
        record: dict = {
            "tag":          tag,
            "n_mel":        n_mel,
            "n_melvecs":    n_melvecs,
            "dropout":      dropout,
            "mixup_alpha":  mixup,
            "aug_profile":  aug,
            "norm_mode":    norm_mode,
            "bg_overlay":   bg_overlay,
            "val_acc":      val_acc,
            "test_acc":     test_acc,
            "packet_bytes": 2 * n_mel * n_melvecs,
            "model_dir":    cfg.MODEL_DIR,
        }
        results.append(record)
        done_tags.add(tag)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if val_acc is not None:
            print(f"  val={val_acc:.4f}  test={test_acc:.4f}"
                  f"  pkt={2*n_mel*n_melvecs}B  → saved to {results_path}")

    # -- final summary -------------------------------------------------------
    n_ok = sum(1 for r in results if r.get("test_acc") is not None)
    print(f"\n  Sweep finished — {n_ok}/{len(results)} runs succeeded.")
    print(f"  Results: {results_path}")

    _plot_results(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--epochs",      default=150,  show_default=True,
              help="Max epochs per run.")
@click.option("--patience",    default=30,   show_default=True,
              help="Early-stopping patience (val_loss).")
@click.option("--output-dir",  default="./big_sweep_results", show_default=True,
              help="Directory for results, model checkpoints, and plots.")
@click.option("--real-data",   default=None,
              help="Override REAL_DATA_DIR (path to recorded .wav files).")
@click.option("--author",      default="sweep", show_default=True,
              help="Author prefix used in W&B run names and model dirs.")
@click.option("--n-mel",       multiple=True, type=int, default=(),
              help="Restrict to specific N_MEL values (repeatable). "
                   "Default: all [20, 24, 28, 32].")
@click.option("--n-melvecs",   multiple=True, type=int, default=(),
              help="Restrict to specific n_frames values (repeatable). "
                   "Default: all [20, 26, 32, 38].")
@click.option("--aug-profile", multiple=True,
              type=click.Choice(["light", "standard", "full"]),
              default=(),
              help="Restrict to specific aug profiles (repeatable). "
                   "Default: all [light, standard, full].")
@click.option("--db-norm", "--db_norm", "norm_mode", flag_value="db", default=True,
              help="Normalise log-mel by fixed dB scaling: /80 → [-1, 0]. (default)")
@click.option("--l2-norm", "--l2_norm", "norm_mode", flag_value="l2",
              help="Normalise log-mel by L2 norm (amplitude-invariant, "
                   "matches h5a2_audio KNN pipeline). Applied to both real and synthetic data.")
@click.option("--zscore-norm", "--zscore_norm", "norm_mode", flag_value="zscore",
              help="Per-band z-score along the time axis. "
                   "Each frequency band is independently centred and scaled to unit variance. "
                   "WARNING: amplifies noise in silent bands.")
@click.option("--mcu-norm", "--mcu_norm", "norm_mode", flag_value="mcu_linear",
              help="MCU-matched linear magnitude mel (no log, Hamming window, /max → [0,1]). "
                   "Simulates spectrogram.c DSP: arm_rfft_q15 → |magnitude| → mel filterbank. "
                   "Use with MCUMatchConfig (HOP_LENGTH=512, N_FFT=512, SR=10200). "
                   "At inference, MCU Q1.15 int16 values are divided by 32768.")
@click.option("--bg-overlay", "--bg_overlay", is_flag=True, default=False,
              help="Enable background overlay augmentation. "
                   "Mixes a random chunk of a real background recording "
                   "into each training sample (waveform domain). "
                   "Requires --bg-dir (or cfg.BACKGROUND_DATA_DIR).")
@click.option("--bg-dir", "--bg_dir", default="",
              help="Path to directory containing background .wav files "
                   "(scanned recursively). Used with --bg-overlay.")
def big_sweep_cmd(
    epochs: int,
    patience: int,
    output_dir: str,
    real_data: str | None,
    author: str,
    n_mel: tuple,
    n_melvecs: tuple,
    aug_profile: tuple,
    norm_mode: str,
    bg_overlay: bool,
    bg_dir: str,
) -> None:
    """
    Hyperparameter sweep: N_MEL × n_frames × dropout × mixup × aug.

    Full grid = 4 × 4 × 3 × 3 × 3 = 432 runs.
    USE_SE_BLOCKS=True is fixed for all runs.

    Saves sweep_results.json after every run (crash-safe).
    Re-running the same --output-dir skips completed combos automatically.

    \b
    Examples:
      # Full sweep with default dB normalisation
      uv run big-sweep --epochs 150 --patience 30

      # Full sweep with L2 norm (hands-on audio style)
      uv run big-sweep --epochs 150 --patience 30 --l2-norm

      # Quick smoke test
      uv run big-sweep --epochs 5 --patience 5 --output-dir /tmp/smoke

      # Sub-grid: only mel=24, all n_frames, standard aug, L2 norm
      uv run big-sweep --n-mel 24 --aug-profile standard --l2-norm

      # Resume after crash (norm_mode is encoded in tags, so it resumes correctly)
      uv run big-sweep --output-dir ./big_sweep_results --l2-norm

      # With background overlay augmentation
      uv run big-sweep --bg-overlay --bg-dir ./audio_files/background
    """
    from classification.datasets import Dataset

    os.makedirs(output_dir, exist_ok=True)

    # Apply CLI filters to build sub-grid
    grid = dict(SWEEP_GRID)
    if n_mel:
        grid["n_mel"]      = [v for v in grid["n_mel"]      if v in n_mel]
    if n_melvecs:
        grid["n_melvecs"]  = [v for v in grid["n_melvecs"]  if v in n_melvecs]
    if aug_profile:
        grid["aug_profile"] = [v for v in grid["aug_profile"] if v in aug_profile]

    total = _total_combos(grid)
    print(f"  Grid: {total} combinations "
          f"(filtered from {_total_combos(SWEEP_GRID)} full grid)")

    dataset = Dataset()
    for cls in ("background", "birds", "handsaw", "helicopter"):
        try:
            dataset.remove_class(cls)
        except KeyError:
            pass

    run_big_sweep(dataset, grid, epochs, patience, output_dir, real_data, author,
                  norm_mode, bg_overlay, bg_dir)

    # Force exit: TF Metal GPU threads and wandb background threads are
    # non-daemon and would otherwise keep the process alive indefinitely.
    os._exit(0)


if __name__ == "__main__":
    big_sweep_cmd()
