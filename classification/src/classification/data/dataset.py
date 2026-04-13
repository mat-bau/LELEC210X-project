"""
Dataset loading and preparation pipeline.

Functions:
    _load_wav_dir         -- load .wav files from a folder/class hierarchy
    load_background_samples -- load background class without augmentation
    _synth_dataset_to_array -- convert legacy Feature_vector_DS to numpy arrays
    prepare_dataset       -- full train/val/test split pipeline
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from ..configs.base_config import BaseConfig
from ..utils.audio_student import AudioUtil, Feature_vector_DS
from .augmentation import AudioAugmentation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SynthDataset(Feature_vector_DS):
    """Bridge between legacy Feature_vector_DS and modern augmentation pipeline."""

    def __init__(self, *args, augment: bool = False, cfg: BaseConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.cfg = cfg or BaseConfig()

    def get_audiosignal(self, idx):
        aud = AudioUtil.open(self.dataset[idx])
        aud = AudioUtil.resample(aud, self.sr)
        if self.augment:
            aud = AudioAugmentation.apply(aud, self.cfg)
        if self.data_aug:
            if "add_bg" in self.data_aug:
                aud = AudioUtil.add_bg(aud, self.dataset, num_sources=1,
                                       max_ms=self.duration, amplitude_limit=0.1)
            if "noise"   in self.data_aug:
                aud = AudioUtil.add_noise(aud, sigma=0.05)
            if "echo"    in self.data_aug:
                aud = AudioUtil.echo(aud)
            if "scaling" in self.data_aug:
                aud = AudioUtil.scaling(aud, scaling_limit=5)
        sig, sr = aud
        return sig / (np.max(np.abs(sig)) + 1e-8), sr

    def __getitem__(self, idx):
        aud = self.get_audiosignal(idx)
        spec = AudioUtil.melspectrogram(aud, Nmel=self.nmel, Nft=self.Nft)
        if self.augment and self.cfg.ENABLE_SPEC_MASKING and np.random.random() > 0.5:
            spec = AudioUtil.spectro_aug_timefreq_masking(
                spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return spec

    def treat_spec(self, spec):
        n_cols = spec.shape[1]
        if n_cols < self.ncol:
            spec = np.pad(spec, ((0, 0), (0, self.ncol - n_cols)), constant_values=0)
            n_cols = self.ncol
        idxs = np.arange(0, n_cols - self.ncol + 1, self.step, dtype=int)
        if len(idxs) == 0:
            idxs = np.array([0])
        windows = []
        for i in idxs:
            w = spec[:, i:i + self.ncol]
            if w.shape[1] < self.ncol:
                w = np.pad(w, ((0, 0), (0, self.ncol - w.shape[1])), constant_values=0)
            windows.append(w)
        fv = np.array(windows).reshape(len(windows), -1)
        if self.normalize:
            norms = np.linalg.norm(fv, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            fv /= norms
        if self.pca is not None:
            fv = np.array([self.pca.transform([v])[0] for v in fv])
        return fv


def _load_wav_dir(folder: str, classnames: list, label_map: dict,
                  extractor, aug_passes: int = 0,
                  subfolder: str = None, skip_noise: bool = False) -> tuple:
    """
    Load all .wav files from folder[/subfolder]/classname/ with optional augmentation.

    Returns (X, y) arrays where X is (N, N_MEL, n_frames, 1) and y is (N,) integer labels.
    """
    X_list, y_list = [], []
    for cls in classnames:
        path = (os.path.join(folder, cls, subfolder)
                if subfolder else os.path.join(folder, cls))
        if not os.path.exists(path):
            print(f"    SKIP  {path}")
            continue
        files = sorted(f for f in os.listdir(path) if f.endswith(".wav"))
        print(f"    {cls:<15} {len(files):>4} files")
        for fname in files:
            fp = os.path.join(path, fname)
            for w in extractor.extract_from_file(fp, augment=False):
                X_list.append(w)
                y_list.append(label_map[cls])
            for _ in range(aug_passes):
                for w in extractor.extract_from_file(fp, augment=True,
                                                     skip_noise=skip_noise):
                    X_list.append(w)
                    y_list.append(label_map[cls])
    return np.array(X_list)[..., np.newaxis], np.array(y_list)


def load_background_samples(cfg: BaseConfig, extractor) -> tuple:
    """
    Load background audio samples with AUG_PASSES=0.

    Expects cfg.BACKGROUND_DATA_DIR to point at a folder containing .wav files.
    Returns (X, y) where y is all-zeros (label index 0 reserved for background
    when classnames[0] == 'background').
    """
    if not cfg.BACKGROUND_DATA_DIR or not os.path.exists(cfg.BACKGROUND_DATA_DIR):
        return np.empty((0,) + extractor.input_shape), np.empty((0,), dtype=int)

    files = sorted(f for f in os.listdir(cfg.BACKGROUND_DATA_DIR)
                   if f.endswith(".wav"))
    X_list = []
    for fname in files:
        fp = os.path.join(cfg.BACKGROUND_DATA_DIR, fname)
        for w in extractor.extract_from_file(fp, augment=False):
            X_list.append(w)

    X = np.array(X_list)[..., np.newaxis] if X_list else \
        np.empty((0,) + extractor.input_shape)
    y = np.zeros(len(X_list), dtype=int)
    print(f"    {'background':<15} {len(files):>4} files  -> {len(X_list)} windows")
    return X, y


def _synth_dataset_to_array(dataset, classnames: list, extractor,
                             augment: bool, cfg: BaseConfig) -> tuple:
    """Convert the legacy Feature_vector_DS synthetic dataset to 4D numpy arrays."""
    label_map = {c: i for i, c in enumerate(classnames)}
    n_mel, n_frames = extractor.cfg.N_MEL, extractor.n_frames
    expected = n_mel * n_frames

    ds = _SynthDataset(
        dataset,
        Nft=cfg.N_FFT, nmel=cfg.N_MEL,
        duration=cfg.DURATION_MS, step=cfg.DURATION_MS // 2,
        augment=augment, cfg=cfg,
    )
    X_flat, y_str = ds.get_feature_vectors()

    X_list, y_list = [], []
    for vec, label in zip(X_flat, y_str):
        if len(vec) == expected:
            window = vec.reshape(n_mel, n_frames)
        else:
            window = np.interp(
                np.linspace(0, len(vec) - 1, expected),
                np.arange(len(vec)), vec,
            ).reshape(n_mel, n_frames)
        X_list.append(window)
        y_list.append(label_map[label])
    return np.array(X_list)[..., np.newaxis], np.array(y_list)


def _print_split_summary(classnames, y_train, y_val, y_test, X_train, X_orig_train):
    SEP, sep = "=" * 70, "-" * 70
    print(f"\n{SEP}\n  DATASET SUMMARY\n{SEP}")
    header = ("  " + f"{'Split':<10} {'Total':>8}  " +
              "  ".join(f"{c[:7]:>8}" for c in classnames))
    print(header)
    print(sep)
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = Counter(y)
        row = "  ".join(f"{counts.get(i, 0):>8}" for i in range(len(classnames)))
        print(f"  {name:<10} {len(y):>8}  {row}")
    print(sep)
    pct = 100 * len(X_orig_train) / max(len(X_train), 1)
    print(f"  Original / augmented in train : {pct:.1f}% / {100 - pct:.1f}%")
    print(SEP)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dataset(synth_dataset, cfg: BaseConfig, extractor,
                    synth_aug_passes: int = None) -> tuple:
    """
    Build train / val / test splits with zero data leakage.

    Split strategy:
        1. Load all raw files (real + synthetic, 0 augmentation).
           If cfg.BACKGROUND_DATA_DIR is set, also load background samples
           as an additional class (index 0, inserted before other classes).
        2. Stratified split on raw pool -> train_raw / val (val stays pure).
        3. Augment train_raw only -> final train set.
        4. Test set is fully isolated from the pool.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, classnames
    """
    if synth_aug_passes is None:
        synth_aug_passes = cfg.SYNTH_AUG_PASSES

    classnames = synth_dataset.list_classes()
    use_background = bool(cfg.BACKGROUND_DATA_DIR)

    # Insert background as class 0 if active
    if use_background and "background" not in classnames:
        classnames = ["background"] + list(classnames)

    label_map = {c: i for i, c in enumerate(classnames)}
    real_subfolders = cfg.REAL_TRAIN_SUBFOLDERS

    SEP, sep = "=" * 70, "-" * 70
    print(f"\n{SEP}\n  DATASET PREPARATION\n{SEP}")

    # -- 1. Real recordings (raw) --------------------------------------------
    real_parts_X, real_parts_y = [], []
    print(f"\n  [1/4] Real recordings  ({cfg.REAL_DATA_DIR})")
    print(f"        Subfolders : {real_subfolders}\n{sep}")
    sound_classes = [c for c in classnames if c != "background"]
    for subfolder in real_subfolders:
        if not any(os.path.exists(os.path.join(cfg.REAL_DATA_DIR, c, subfolder))
                   for c in sound_classes):
            print(f"  SKIP  {subfolder}/ not found")
            continue
        X_sub, y_sub = _load_wav_dir(
            cfg.REAL_DATA_DIR, sound_classes, label_map, extractor,
            aug_passes=0, subfolder=subfolder, skip_noise=True,
        )
        real_parts_X.append(X_sub)
        real_parts_y.append(y_sub)

    if real_parts_X:
        X_real = np.concatenate(real_parts_X, axis=0)
        y_real = np.concatenate(real_parts_y, axis=0)
        counts = Counter(y_real)
        print(f"{sep}\n  Real loaded  : {len(X_real)} feature vectors")
        for i, c in enumerate(classnames):
            if counts.get(i, 0):
                print(f"    {c:<15} {counts[i]:>5}")
    else:
        X_real = np.empty((0, *extractor.input_shape))
        y_real = np.empty((0,), dtype=int)

    # -- Background samples --------------------------------------------------
    if use_background:
        print(f"\n  Background samples  ({cfg.BACKGROUND_DATA_DIR})\n{sep}")
        X_bg, y_bg = load_background_samples(cfg, extractor)
    else:
        X_bg = np.empty((0, *extractor.input_shape))
        y_bg = np.empty((0,), dtype=int)

    # -- 2. Synthetic data (raw) ---------------------------------------------
    print(f"\n  [2/4] Synthetic data (0 augmentation)\n{sep}")
    X_synth, y_synth = _synth_dataset_to_array(
        synth_dataset, sound_classes, extractor, augment=False, cfg=cfg)
    # Re-label synth to match updated label_map (background shifts indices)
    y_synth = np.array([label_map[sound_classes[yi]] for yi in y_synth])
    counts = Counter(y_synth)
    print(f"  Synth loaded : {len(X_synth)} feature vectors")
    for i, c in enumerate(classnames):
        if counts.get(i, 0):
            print(f"    {c:<15} {counts[i]:>5}")

    # -- 3. Stratified split on raw pool -------------------------------------
    print(f"\n  [3/4] Stratified split "
          f"{int(cfg.TRAIN_SPLIT_RATIO*100)}/{int((1-cfg.TRAIN_SPLIT_RATIO)*100)}\n{sep}")

    all_X_parts = [X_synth]
    all_y_parts = [y_synth]
    if real_parts_X:
        all_X_parts += real_parts_X
        all_y_parts += real_parts_y
    if use_background and len(X_bg):
        all_X_parts.append(X_bg)
        all_y_parts.append(y_bg)

    X_pool = np.concatenate(all_X_parts, axis=0)
    y_pool = np.concatenate(all_y_parts, axis=0)
    assert X_pool.ndim == 4, f"Expected 4D pool, got {X_pool.shape}"

    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_pool, y_pool,
        test_size=1 - cfg.TRAIN_SPLIT_RATIO,
        stratify=y_pool,
        random_state=cfg.RANDOM_SEED,
    )
    print(f"  Pool total   : {len(X_pool)} fv  {X_pool.shape[1:]}")
    print(f"  -> Train raw : {len(X_train_raw)}")
    print(f"  -> Val (pure): {len(X_val)}")

    # -- 4. Augmentation on train partition only -----------------------------
    print(f"\n  [4/4] Augmentation (train only)\n{sep}")
    aug_X = [X_train_raw]
    aug_y = [y_train_raw]

    if synth_aug_passes > 0:
        print(f"  Synthetic : {synth_aug_passes} passes")
        for i in range(synth_aug_passes):
            print(f"    pass {i+1:>3}/{synth_aug_passes}...", end="\r")
            Xp, yp = _synth_dataset_to_array(
                synth_dataset, sound_classes, extractor, augment=True, cfg=cfg)
            yp = np.array([label_map[sound_classes[yi]] for yi in yp])
            aug_X.append(Xp)
            aug_y.append(yp)
        print(f"  Synthetic : {synth_aug_passes} passes OK" + " " * 20)
    else:
        print(f"  Synthetic : 0 passes (SYNTH_AUG_PASSES=0)")

    if cfg.REAL_AUG_PASSES > 0 and real_parts_X:
        print(f"  Real      : {cfg.REAL_AUG_PASSES} passes  {real_subfolders}")
        aug_real_X, aug_real_y = [], []
        for subfolder in real_subfolders:
            for cls in sound_classes:
                path = os.path.join(cfg.REAL_DATA_DIR, cls, subfolder)
                if not os.path.exists(path):
                    continue
                for fname in sorted(os.listdir(path)):
                    if not fname.endswith(".wav"):
                        continue
                    fp = os.path.join(path, fname)
                    for _ in range(cfg.REAL_AUG_PASSES):
                        for w in extractor.extract_from_file(fp, augment=True,
                                                             skip_noise=True):
                            aug_real_X.append(w)
                            aug_real_y.append(label_map[cls])
        if aug_real_X:
            X_real_aug = np.array(aug_real_X)[..., np.newaxis]
            y_real_aug = np.array(aug_real_y)
            aug_X.append(X_real_aug)
            aug_y.append(y_real_aug)
            counts = Counter(y_real_aug)
            print(f"  Real aug  : {len(X_real_aug)} fv generated")
            for i, c in enumerate(classnames):
                if counts.get(i, 0):
                    print(f"    {c:<15} {counts.get(i, 0):>5}")
        else:
            print(f"  WARN  no .wav files found in {real_subfolders}")
    else:
        print(f"  Real      : 0 passes")

    X_train = np.concatenate(aug_X, axis=0).astype(np.float32)
    y_train = np.concatenate(aug_y, axis=0)
    X_val = X_val.astype(np.float32)

    # -- Test set (fully isolated) -------------------------------------------
    print(f"\n  Test set  ({cfg.REAL_DATA_DIR}/.../test/)\n{sep}")
    X_test, y_test = _load_wav_dir(
        cfg.REAL_DATA_DIR, sound_classes, label_map, extractor,
        aug_passes=0, subfolder="test",
    )
    X_test = X_test.astype(np.float32)

    _print_split_summary(classnames, y_train, y_val, y_test, X_train, X_train_raw)
    return X_train, X_val, X_test, y_train, y_val, y_test, classnames
