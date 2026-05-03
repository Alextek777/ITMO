"""
=============================================================================
MILimbEEG — Dataset Loader
=============================================================================
Loads raw CSV trials for Motor (M) activity across all subjects.

File naming convention
----------------------
  S{subj}R{rep}{type}{label}_{trial}.csv
  Example: S1R1M6_3.csv  →  Subject 1, Repetition 1, Motor, Label 6, Trial 3

Returned arrays
---------------
  X : np.ndarray  shape (n_trials, n_channels, n_samples) = (N, 16, 500)
  y : np.ndarray  shape (n_trials,)  — class indices 0-7
  meta : list[dict]  — per-trial metadata {subject, rep, label, trial, path}
=============================================================================
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project root is two levels up from this file
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    DATA_DIR, N_CHANNELS, N_SAMPLES,
    LABEL_TO_IDX, LABEL_MAP,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
_PATTERN = re.compile(
    r"S(?P<subj>\d+)R(?P<rep>\d+)(?P<type>[MI])(?P<label>\d+)_(?P<trial>\d+)\.csv"
)


def parse_filename(path: Path) -> Optional[dict]:
    """Extract metadata from a MILimbEEG CSV filename."""
    m = _PATTERN.match(path.name)
    if m is None:
        return None
    return {
        "subject": int(m.group("subj")),
        "rep":     int(m.group("rep")),
        "type":    m.group("type"),    # 'M' or 'I'
        "label":   int(m.group("label")),
        "trial":   int(m.group("trial")),
        "path":    path,
    }


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(
    data_dir:        Path         = DATA_DIR,
    subjects:        Optional[list[int]] = None,
    activity_type:   str          = "M",          # 'M' Motor | 'I' Imagery
    labels:          Optional[list[int]] = None,  # None → all labels
    normalize:       bool         = True,          # z-score per channel per trial
    verbose:         bool         = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Load MILimbEEG CSV files and return (X, y, meta).

    Parameters
    ----------
    data_dir      : root directory of the dataset
    subjects      : list of subject IDs to include (None = all 1-60)
    activity_type : 'M' for Motor, 'I' for Imagery
    labels        : list of integer labels to include (None = all)
    normalize     : z-score normalise each channel independently
    verbose       : show tqdm progress bar

    Returns
    -------
    X    : (N, 16, 500)  float32
    y    : (N,)          int32   — class indices 0-based (label - 1)
    meta : list of N dicts with trial metadata
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Collect all CSV files
    all_csv = sorted(data_dir.rglob("*.csv"))
    log.info(f"Total CSV files found: {len(all_csv)}")

    # Parse & filter
    records: list[dict] = []
    for p in all_csv:
        meta = parse_filename(p)
        if meta is None:
            continue
        if meta["type"] != activity_type:
            continue
        if subjects is not None and meta["subject"] not in subjects:
            continue
        if labels is not None and meta["label"] not in labels:
            continue
        records.append(meta)

    if not records:
        raise ValueError(
            f"No files matched: type={activity_type}, subjects={subjects}, labels={labels}"
        )

    log.info(f"Matched {len(records)} trials")

    # Load arrays
    Xs, ys, metas = [], [], []
    iterator = tqdm(records, desc="Loading trials", disable=not verbose)

    for rec in iterator:
        try:
            df = pd.read_csv(rec["path"], index_col=0)
        except Exception as e:
            log.warning(f"Could not read {rec['path']}: {e}")
            continue

        if df.shape != (N_SAMPLES, N_CHANNELS):
            log.warning(
                f"Unexpected shape {df.shape} in {rec['path'].name}, skipping"
            )
            continue

        x = df.values.T.astype(np.float32)   # → (16, 500)

        if normalize:
            mu  = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-8
            x   = (x - mu) / std

        Xs.append(x)
        ys.append(LABEL_TO_IDX[rec["label"]])
        metas.append(rec)

    X = np.stack(Xs).astype(np.float32)   # (N, 16, 500)
    y = np.array(ys, dtype=np.int32)

    log.info(f"Loaded dataset: X={X.shape}, y={y.shape}")
    _log_class_distribution(y)

    return X, y, metas


# ---------------------------------------------------------------------------
# Cross-subject split helper
# ---------------------------------------------------------------------------

def load_cross_subject_splits(
    data_dir:      Path = DATA_DIR,
    activity_type: str  = "M",
    normalize:     bool = True,
    verbose:       bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict]]]:
    """
    Load train / val / test splits using the predefined subject lists.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing (X, y, meta).
    """
    splits = {}
    split_map = {
        "train": TRAIN_SUBJECTS,
        "val":   VAL_SUBJECTS,
        "test":  TEST_SUBJECTS,
    }
    for split_name, subj_list in split_map.items():
        if verbose:
            print(f"\n[{split_name.upper()}]  subjects: {subj_list[:5]}{'...' if len(subj_list) > 5 else ''}")
        X, y, meta = load_dataset(
            data_dir=data_dir,
            subjects=subj_list,
            activity_type=activity_type,
            normalize=normalize,
            verbose=verbose,
        )
        splits[split_name] = (X, y, meta)

    return splits


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_class_distribution(y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    lines = []
    for idx, cnt in zip(unique, counts):
        label  = idx + 1
        name   = LABEL_MAP.get(label, "?")
        pct    = 100 * cnt / len(y)
        lines.append(f"  class {idx} ({name:4s}): {cnt:5d}  ({pct:.1f}%)")
    log.info("Class distribution:\n" + "\n".join(lines))


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Inverse-frequency class weights for imbalanced training.
    Returned dict maps class index → weight (compatible with Keras).
    """
    from sklearn.utils.class_weight import compute_class_weight
    unique = np.unique(y)
    weights = compute_class_weight("balanced", classes=unique, y=y)
    return dict(zip(unique.tolist(), weights.tolist()))


def get_dataset_summary(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame summarising trial counts per subject × label."""
    records = []
    for p in sorted(data_dir.rglob("*.csv")):
        meta = parse_filename(p)
        if meta is not None:
            records.append(meta)

    df = pd.DataFrame(records).drop(columns=["path"])
    summary = (
        df[df["type"] == "M"]
        .groupby(["subject", "label"])
        .size()
        .unstack(fill_value=0)
    )
    summary.columns = [f"M{c}" for c in summary.columns]
    return summary
