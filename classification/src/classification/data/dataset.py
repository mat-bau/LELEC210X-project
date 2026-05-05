"""
Dataset loading and preparation pipeline.

Split strategy (file-level, zero data leakage):
    Raw .wav files are split 80/20 at the *file* level, not the window level.
    This prevents any two windows from the same recording from appearing in
    both train and val — which was the source of the inflated val_acc.

    1. Collect .wav file paths per class from REAL_TRAIN_SUBFOLDERS.
    2. Stratified file-level split → train_files / val_files (per class).
    3. Val  = raw windows from val_files only. No augmentation, no synthetic.
    4. Train = raw + augmented windows from train_files + all synthetic data.
    5. Test = loaded from subfolder="test" (fully isolated, unchanged).

Functions:
    _load_wav_dir           -- load .wav files into (X, y) arrays  [test only]
    load_background_samples -- load background class without augmentation
    _synth_dataset_to_array -- convert legacy Feature_vector_DS to numpy arrays
    prepare_dataset         -- full train/val/test split pipeline
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

    def __init__(self, *args, augment: bool = False, cfg: BaseConfig = None,
                 norm_mode: str = "db", **kwargs):
        # Always disable the parent's built-in L2 normalisation;
        # treat_spec() below handles all NORM_MODE variants explicitly.
        kwargs.pop("normalize", None)
        super().__init__(*args, normalize=False, **kwargs)
        self.augment = augment
        self.cfg = cfg or BaseConfig()
        self.norm_mode = norm_mode

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

        if self.norm_mode == "mcu_linear":
            # Synth data is in log10(magnitude) from audio_student.py.
            # Convert back to linear magnitude then normalise to [0, 1] per window,
            # matching what _compute_mel("mcu_linear") produces for real .wav files.
            normed = []
            for w in windows:
                lin = np.power(10.0, w)          # log10 → linear magnitude
                lin = lin / (lin.max() + 1e-8)   # → [0, 1]
                normed.append(lin)
            fv = np.array(normed).reshape(len(windows), -1)
        elif self.norm_mode == "zscore":
            # Per-band z-score along time axis — applied per window, per frequency band.
            normed = []
            for w in windows:
                mu = w.mean(axis=1, keepdims=True)
                sigma = w.std(axis=1, keepdims=True)
                normed.append((w - mu) / (sigma + 1e-8))
            fv = np.array(normed).reshape(len(windows), -1)
        else:
            fv = np.array(windows).reshape(len(windows), -1)
            if self.norm_mode == "l2":
                norms = np.linalg.norm(fv, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                fv /= norms
            # "db" mode: no normalisation on synthetic data (log10 scale preserved)

        if self.pca is not None:
            fv = np.array([self.pca.transform([v])[0] for v in fv])
        return fv


def _load_wav_dir(folder: str, classnames: list, label_map: dict,
                  extractor, aug_passes: int = 0,
                  subfolder: str = None, skip_noise: bool = False) -> tuple:
    """
    Load all .wav files from folder[/subfolder]/classname/ with optional augmentation.

    Returns (X, y) arrays where X is (N, N_MEL, n_frames, 1) and y is (N,) integer labels.
    Used exclusively for the test set.
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
        norm_mode=getattr(cfg, "NORM_MODE", "db"),
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


def compute_global_stats(X_train: np.ndarray) -> tuple:
    """
    Compute per-frequency (per-mel-band) mean and std over the entire training set.

    Used for global normalisation so that training and MCU-inference share
    the same statistics (rather than per-sample, which the MCU cannot replicate).

    :param X_train: (N, N_MEL, n_frames, 1) float32 array
    :returns: (mean, std) each of shape (N_MEL,)
    """
    N, n_mel, n_frames, _ = X_train.shape
    flat = X_train[:, :, :, 0].transpose(0, 2, 1).reshape(-1, n_mel)  # (N*T, N_MEL)
    mean = flat.mean(axis=0)   # (N_MEL,)
    std  = flat.std(axis=0)    # (N_MEL,)
    return mean, std


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
    print(f"  Val source : real files only (file-level split, no window leakage)")
    print(SEP)


def _collect_real_files(real_data_dir: str, classnames: list,
                        real_subfolders: list) -> dict:
    """
    Collect all .wav file paths per class from the training subfolders.

    :returns: {cls: [filepath, ...]}
    """
    files_by_class = {}
    for cls in classnames:
        fps = []
        for subfolder in real_subfolders:
            path = os.path.join(real_data_dir, cls, subfolder)
            if os.path.exists(path):
                fps.extend(
                    os.path.join(path, f)
                    for f in sorted(os.listdir(path))
                    if f.endswith(".wav")
                )
        files_by_class[cls] = fps
    return files_by_class


def _file_level_split(files_by_class: dict, test_size: float,
                      random_seed: int) -> tuple[dict, dict]:
    """
    Split file paths per class into train/val at the *file* level.

    Classes with only 1 file get that file in train (with a warning).

    :returns: (train_files_by_class, val_files_by_class)
    """
    train_dict, val_dict = {}, {}
    for cls, fps in files_by_class.items():
        if len(fps) == 0:
            train_dict[cls] = []
            val_dict[cls]   = []
        elif len(fps) == 1:
            print(f"  WARN  {cls}: only 1 recording file — "
                  f"put entirely in train, no val contribution")
            train_dict[cls] = list(fps)
            val_dict[cls]   = []
        else:
            tr, va = train_test_split(
                fps, test_size=test_size, random_state=random_seed,
            )
            train_dict[cls] = tr
            val_dict[cls]   = va
    return train_dict, val_dict


def _windows_from_file_list(fps: list, label: int, extractor,
                             aug_passes: int = 0,
                             skip_noise: bool = False) -> tuple:
    """
    Extract sliding-window mel-spectrograms from a list of .wav files.

    :returns: (X_list, y_list) as flat Python lists (not yet numpy)
    """
    X, y = [], []
    for fp in fps:
        # Raw pass (always)
        for w in extractor.extract_from_file(fp, augment=False):
            X.append(w)
            y.append(label)
        # Augmented passes
        for _ in range(aug_passes):
            for w in extractor.extract_from_file(fp, augment=True,
                                                  skip_noise=skip_noise):
                X.append(w)
                y.append(label)
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dataset(synth_dataset, cfg: BaseConfig, extractor,
                    synth_aug_passes: int = None) -> tuple:
    """
    Build train / val / test splits with zero data leakage.

    Split strategy (file-level):
        - For each sound class, raw .wav files are split 80/20 at the file level.
        - Val set  = raw windows from the 20% val files (no augmentation, no synthetic).
        - Train set = raw + augmented windows from 80% train files + all synthetic.
        - Test set  = loaded from subfolder "test/" (always fully isolated).

        No two windows from the same recording can appear in both train and val.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, classnames
    """
    if synth_aug_passes is None:
        synth_aug_passes = cfg.SYNTH_AUG_PASSES

    classnames = synth_dataset.list_classes()
    # Load background as a training CLASS only when ENABLE_BG_OVERLAY is False.
    # When bg-overlay is active, BACKGROUND_DATA_DIR is used solely as an
    # augmentation noise source — not as a separate output class.
    use_background = (bool(cfg.BACKGROUND_DATA_DIR)
                      and not getattr(cfg, "ENABLE_BG_OVERLAY", False))

    if use_background and "background" not in classnames:
        classnames = ["background"] + list(classnames)

    label_map = {c: i for i, c in enumerate(classnames)}
    sound_classes = [c for c in classnames if c != "background"]
    real_subfolders = cfg.REAL_TRAIN_SUBFOLDERS

    SEP, sep = "=" * 70, "-" * 70
    print(f"\n{SEP}\n  DATASET PREPARATION  (file-level split)\n{SEP}")

    # ------------------------------------------------------------------ #
    # 1. Collect real file paths per class                                 #
    # ------------------------------------------------------------------ #
    print(f"\n  [1/4] Collecting real recordings  ({cfg.REAL_DATA_DIR})")
    print(f"        Subfolders : {real_subfolders}\n{sep}")

    files_by_class = _collect_real_files(cfg.REAL_DATA_DIR, sound_classes,
                                          real_subfolders)
    for cls, fps in files_by_class.items():
        print(f"    {cls:<15} {len(fps):>4} files")

    # Background files (for Task 3)
    bg_all_files: list = []
    if use_background and cfg.BACKGROUND_DATA_DIR and \
            os.path.exists(cfg.BACKGROUND_DATA_DIR):
        bg_all_files = [
            os.path.join(cfg.BACKGROUND_DATA_DIR, f)
            for f in sorted(os.listdir(cfg.BACKGROUND_DATA_DIR))
            if f.endswith(".wav")
        ]
        print(f"    {'background':<15} {len(bg_all_files):>4} files")

    # ------------------------------------------------------------------ #
    # 2. File-level stratified split                                       #
    # ------------------------------------------------------------------ #
    print(f"\n  [2/4] File-level split "
          f"{int(cfg.TRAIN_SPLIT_RATIO*100)}/{int((1-cfg.TRAIN_SPLIT_RATIO)*100)}\n{sep}")

    train_files, val_files = _file_level_split(
        files_by_class,
        test_size=1 - cfg.TRAIN_SPLIT_RATIO,
        random_seed=cfg.RANDOM_SEED,
    )
    for cls in sound_classes:
        print(f"    {cls:<15}  train {len(train_files[cls]):>3} files"
              f"  |  val {len(val_files[cls]):>3} files")

    # Background split
    bg_train_files: list = []
    bg_val_files:   list = []
    if bg_all_files:
        if len(bg_all_files) >= 2:
            bg_train_files, bg_val_files = train_test_split(
                bg_all_files,
                test_size=1 - cfg.TRAIN_SPLIT_RATIO,
                random_state=cfg.RANDOM_SEED,
            )
        else:
            bg_train_files = list(bg_all_files)
        print(f"    {'background':<15}  train {len(bg_train_files):>3} files"
              f"  |  val {len(bg_val_files):>3} files")

    # ------------------------------------------------------------------ #
    # 3. Val set: raw windows from val files (no augmentation, no synth)   #
    # ------------------------------------------------------------------ #
    print(f"\n  [3/4] Building val set (raw windows from val files)\n{sep}")

    X_val_list: list = []
    y_val_list: list = []
    for cls in sound_classes:
        xv, yv = _windows_from_file_list(val_files[cls], label_map[cls], extractor)
        X_val_list.extend(xv)
        y_val_list.extend(yv)
        print(f"    {cls:<15} {len(val_files[cls]):>3} files → {len(xv):>5} windows")

    if bg_val_files:
        xv, yv = _windows_from_file_list(bg_val_files, label_map["background"],
                                          extractor)
        X_val_list.extend(xv)
        y_val_list.extend(yv)
        print(f"    {'background':<15} {len(bg_val_files):>3} files → {len(xv):>5} windows")

    if X_val_list:
        X_val = np.array(X_val_list)[..., np.newaxis].astype(np.float32)
        y_val = np.array(y_val_list, dtype=int)
    else:
        X_val = np.empty((0, *extractor.input_shape), dtype=np.float32)
        y_val = np.empty((0,), dtype=int)
        print("  WARN  val set is empty — all classes had ≤1 recording file")

    print(f"  Val total : {len(y_val)} windows")

    # ------------------------------------------------------------------ #
    # 4. Train set: real (raw + aug) + synthetic                           #
    # ------------------------------------------------------------------ #
    print(f"\n  [4/4] Building train set\n{sep}")

    X_train_parts: list = []
    y_train_parts: list = []

    # -- Real recordings (train files, raw + augmented) --
    print(f"  Real raw (train files):")
    n_real_raw = 0
    for cls in sound_classes:
        xt, yt = _windows_from_file_list(train_files[cls], label_map[cls], extractor)
        X_train_parts.extend(xt)
        y_train_parts.extend(yt)
        n_real_raw += len(xt)
        print(f"    {cls:<15} {len(train_files[cls]):>3} files → {len(xt):>5} windows")

    if bg_train_files:
        xt, yt = _windows_from_file_list(bg_train_files, label_map["background"],
                                          extractor)
        X_train_parts.extend(xt)
        y_train_parts.extend(yt)
        n_real_raw += len(xt)
        print(f"    {'background':<15} {len(bg_train_files):>3} files → {len(xt):>5} windows")

    # -- Real augmented passes --
    if cfg.REAL_AUG_PASSES > 0:
        print(f"  Real aug ({cfg.REAL_AUG_PASSES} passes per file):")
        n_aug_before = len(X_train_parts)
        for cls in sound_classes:
            fps = train_files[cls]
            if not fps:
                continue
            label = label_map[cls]
            for _ in range(cfg.REAL_AUG_PASSES):
                for fp in fps:
                    for w in extractor.extract_from_file(fp, augment=True,
                                                          skip_noise=True):
                        X_train_parts.append(w)
                        y_train_parts.append(label)
        if bg_train_files:
            for _ in range(cfg.REAL_AUG_PASSES):
                for fp in bg_train_files:
                    for w in extractor.extract_from_file(fp, augment=True,
                                                          skip_noise=True):
                        X_train_parts.append(w)
                        y_train_parts.append(label_map["background"])
        n_aug = len(X_train_parts) - n_aug_before
        print(f"  Real aug  : {n_aug} windows generated")
    else:
        print(f"  Real aug  : 0 passes (REAL_AUG_PASSES=0)")

    # Convert real portions to numpy
    if X_train_parts:
        X_train_real = np.array(X_train_parts)[..., np.newaxis].astype(np.float32)
        y_train_real = np.array(y_train_parts, dtype=int)
    else:
        X_train_real = np.empty((0, *extractor.input_shape), dtype=np.float32)
        y_train_real = np.empty((0,), dtype=int)

    # -- Synthetic data (all in train) --
    print(f"\n  Synthetic (0 augmentation):\n{sep}")
    X_synth, y_synth = _synth_dataset_to_array(
        synth_dataset, sound_classes, extractor, augment=False, cfg=cfg)
    y_synth = np.array([label_map[sound_classes[yi]] for yi in y_synth])
    counts = Counter(y_synth.tolist())
    print(f"  Synth raw : {len(X_synth)} windows")
    for i, c in enumerate(classnames):
        if counts.get(i, 0):
            print(f"    {c:<15} {counts[i]:>5}")

    all_X = [X_synth, X_train_real]
    all_y = [y_synth, y_train_real]

    if synth_aug_passes and synth_aug_passes > 0:
        print(f"  Synthetic aug : {synth_aug_passes} passes")
        for i in range(synth_aug_passes):
            print(f"    pass {i+1:>3}/{synth_aug_passes}...", end="\r")
            Xp, yp = _synth_dataset_to_array(
                synth_dataset, sound_classes, extractor, augment=True, cfg=cfg)
            yp = np.array([label_map[sound_classes[yi]] for yi in yp])
            all_X.append(Xp)
            all_y.append(yp)
        print(f"  Synthetic aug : {synth_aug_passes} passes OK" + " " * 20)
    else:
        print(f"  Synthetic aug : 0 passes (SYNTH_AUG_PASSES=0)")

    X_train = np.concatenate(all_X, axis=0).astype(np.float32)
    y_train = np.concatenate(all_y, axis=0)

    # ------------------------------------------------------------------ #
    # 5. Test set (always fully isolated by subfolder="test")              #
    # ------------------------------------------------------------------ #
    print(f"\n  Test set  ({cfg.REAL_DATA_DIR}/.../test/)\n{sep}")
    X_test, y_test = _load_wav_dir(
        cfg.REAL_DATA_DIR, sound_classes, label_map, extractor,
        aug_passes=0, subfolder="test",
    )
    X_test = X_test.astype(np.float32)

    # X_orig_train = real raw portion (for summary percentage)
    X_orig_train = X_train_real[:n_real_raw] if n_real_raw else X_train_real

    _print_split_summary(classnames, y_train, y_val, y_test, X_train, X_orig_train)
    return X_train, X_val, X_test, y_train, y_val, y_test, classnames
