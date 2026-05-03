"""
=============================================================================
MILimbEEG — Preprocessing Pipeline
=============================================================================
Transforms raw EEG epochs (N, 16, 500) into model-ready tensors.

Steps
-----
1. Band-pass filter  [1–49 Hz]  via MNE (zero-phase)
2. Common Average Re-reference (CAR)
3. Z-score normalisation per channel per epoch
4. Optional: Exponential Moving Standardization (online baseline)
5. Reshape for model input:
   - EEGNet  : (N, 1, 16, 500)   ← 4-D "image" for Conv2D
   - ResNet  : (N, 500, 16)       ← sequence-first for Conv1D
   - BiLSTM  : (N, 500, 16)       ← same

Data augmentation (training only)
----------------------------------
- Gaussian noise injection
- Channel dropout (random channels zeroed)
- Time-shift (random circular roll)
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import mne

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import SFREQ, N_CHANNELS, N_SAMPLES, CH_NAMES, CH_TYPES, RANDOM_SEED

log = logging.getLogger(__name__)
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# MNE Info object (reused across calls)
# ---------------------------------------------------------------------------

def _make_mne_info() -> mne.Info:
    return mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=CH_TYPES)


# ---------------------------------------------------------------------------
# Signal-level transforms
# ---------------------------------------------------------------------------

def bandpass_filter(
    X: np.ndarray,
    l_freq: float = 1.0,
    h_freq: float = 49.0,
    sfreq:  float = SFREQ,
) -> np.ndarray:
    """
    Zero-phase FIR band-pass filter applied via MNE.

    Parameters
    ----------
    X     : (N, C, T)  raw epochs
    l_freq: low-frequency cut-off in Hz
    h_freq: high-frequency cut-off in Hz

    Returns
    -------
    X_filt : (N, C, T)  filtered array
    """
    info = _make_mne_info()
    epochs_array = mne.EpochsArray(X, info, verbose=False)
    epochs_filt  = epochs_array.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return epochs_filt.get_data().astype(np.float32)


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """
    Subtract the mean across all channels (CAR) per sample.

    Parameters
    ----------
    X : (N, C, T)

    Returns
    -------
    X_car : (N, C, T)
    """
    mean_ref = X.mean(axis=1, keepdims=True)   # (N, 1, T)
    return (X - mean_ref).astype(np.float32)


def zscore_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score each channel independently over the time axis.

    Parameters
    ----------
    X : (N, C, T)

    Returns
    -------
    X_norm : (N, C, T)
    """
    mu  = X.mean(axis=2, keepdims=True)         # (N, C, 1)
    std = X.std(axis=2,  keepdims=True) + eps   # (N, C, 1)
    return ((X - mu) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    X:        np.ndarray,
    do_filter: bool = True,
    do_car:    bool = True,
    do_zscore: bool = True,
    l_freq:    float = 1.0,
    h_freq:    float = 49.0,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline in one call.

    Parameters
    ----------
    X         : (N, 16, 500)  raw epochs
    do_filter : apply MNE band-pass filter
    do_car    : apply common average reference
    do_zscore : apply per-channel z-score normalisation

    Returns
    -------
    X_proc : (N, 16, 500)  preprocessed array
    """
    if do_filter:
        log.info(f"Band-pass filtering [{l_freq}–{h_freq} Hz] ...")
        X = bandpass_filter(X, l_freq=l_freq, h_freq=h_freq)

    if do_car:
        log.info("Applying Common Average Reference ...")
        X = common_average_reference(X)

    if do_zscore:
        log.info("Z-score normalising ...")
        X = zscore_normalize(X)

    return X


# ---------------------------------------------------------------------------
# Model-specific reshape helpers
# ---------------------------------------------------------------------------

def to_eegnet_format(X: np.ndarray) -> np.ndarray:
    """
    Reshape to EEGNet Conv2D format (channels_last, CPU-compatible).

    Input  : (N, C, T)
    Output : (N, C, T, 1)   ← height=C, width=T, depth=1 (single colour)

    This layout lets TF's CPU Conv2D (NHWC only) treat:
      height = EEG electrodes (16)
      width  = time samples   (500)
      channels = 1
    """
    return X[:, :, :, np.newaxis]     # (N, 16, 500, 1)


def to_sequence_format(X: np.ndarray) -> np.ndarray:
    """
    Reshape to sequence format for Conv1D / LSTM models.

    Input  : (N, C, T)
    Output : (N, T, C)   ← time-first
    """
    return np.transpose(X, (0, 2, 1))  # (N, 500, 16)


# ---------------------------------------------------------------------------
# Data augmentation (training only)
# ---------------------------------------------------------------------------

def augment(
    X:            np.ndarray,
    y:            np.ndarray,
    noise_std:    float = 0.05,
    time_shift_p: float = 0.3,
    ch_drop_p:    float = 0.2,
    max_shift:    int   = 25,     # samples  (0.2 s at 125 Hz)
    rng:          np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    In-place augmentation applied to training batches.

    Augmentations
    -------------
    1. Additive Gaussian noise      (always, std=noise_std)
    2. Random circular time shift   (prob time_shift_p)
    3. Random channel dropout       (prob ch_drop_p per channel)

    Parameters
    ----------
    X         : (N, C, T) float32
    y         : (N,)      int32
    noise_std : standard deviation of Gaussian noise
    time_shift_p : probability of applying time shift to a sample
    ch_drop_p    : probability of dropping (zeroing) a channel per sample
    max_shift    : maximum |shift| in samples

    Returns
    -------
    X_aug, y  — augmented data (same y, augmented X)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    X_aug = X.copy()
    N, C, T = X_aug.shape

    # 1. Gaussian noise
    X_aug += rng.normal(0, noise_std, size=X_aug.shape).astype(np.float32)

    # 2. Time shift
    shift_mask = rng.random(N) < time_shift_p
    for i in np.where(shift_mask)[0]:
        shift = rng.integers(-max_shift, max_shift + 1)
        X_aug[i] = np.roll(X_aug[i], shift, axis=1)

    # 3. Channel dropout
    drop_mask = rng.random((N, C)) < ch_drop_p
    X_aug[drop_mask] = 0.0

    return X_aug, y


# ---------------------------------------------------------------------------
# Utility — convert raw NumPy arrays to mne.EpochsArray for visualisation
# ---------------------------------------------------------------------------

def to_mne_epochs(
    X: np.ndarray,
    y: np.ndarray,
    event_id: dict[str, int] | None = None,
) -> mne.EpochsArray:
    """
    Wrap a NumPy array as an MNE EpochsArray for visualisation & inspection.

    Parameters
    ----------
    X        : (N, C, T)   — must be in µV or already normalised
    y        : (N,)  class indices 0-based
    event_id : dict mapping class name → code (uses LABEL_MAP if None)

    Returns
    -------
    mne.EpochsArray
    """
    from config import LABEL_MAP

    if event_id is None:
        event_id = {v: k for k, v in LABEL_MAP.items()}

    info = _make_mne_info()

    # MNE expects events array: (N, 3) with columns [sample, 0, event_code]
    event_codes = y + 1   # shift back to 1-based label for MNE event_id lookup
    events = np.column_stack([
        np.arange(len(y)),
        np.zeros(len(y), dtype=int),
        event_codes,
    ])

    return mne.EpochsArray(X, info, events=events, event_id=event_id, verbose=False)
