"""
=============================================================================
EEG Motor Classification — Visualization Utilities
=============================================================================
Provides publication-quality plots for:
  - Raw / filtered EEG signals
  - MNE topomaps & power spectra
  - Class-conditional time-frequency maps (ERSP)
  - Training curves (loss / accuracy)
  - Confusion matrices
  - Per-class ROC & Precision-Recall curves
  - Attention weight maps (Bi-LSTM)
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import mne

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    SFREQ, CH_NAMES, CLASS_NAMES, N_CLASSES,
    LABEL_MAP, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

PALETTE    = sns.color_palette("tab10", N_CLASSES)
FIGSIZE_SM = (10, 4)
FIGSIZE_MD = (14, 6)
FIGSIZE_LG = (16, 10)

matplotlib.rcParams.update({
    "figure.dpi":     120,
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Signal-level plots
# ---------------------------------------------------------------------------

def plot_raw_trial(
    x:         np.ndarray,
    label:     int,
    channels:  list[str]  = CH_NAMES,
    sfreq:     float       = SFREQ,
    figsize:   tuple       = (14, 10),
    title:     str | None  = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot all 16 EEG channels of a single trial.

    Parameters
    ----------
    x       : (C, T) array — one EEG trial
    label   : integer class label (1-based)
    """
    C, T   = x.shape
    times  = np.arange(T) / sfreq
    action = LABEL_MAP.get(label, str(label))

    fig, axes = plt.subplots(C, 1, figsize=figsize, sharex=True)
    fig.suptitle(title or f"EEG Trial  |  Action: {action}  (label={label})",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        ax.plot(times, x[i], linewidth=0.8, color=f"C{i % 10}")
        ax.set_ylabel(ch, rotation=0, labelpad=30, va="center", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(left=False, labelleft=False)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_channel_psd(
    X:         np.ndarray,
    y:         np.ndarray,
    channel:   int        = 10,    # C3 — primary motor cortex
    sfreq:     float      = SFREQ,
    fmin:      float      = 1.0,
    fmax:      float      = 50.0,
    figsize:   tuple      = FIGSIZE_MD,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot mean PSD per class for a single EEG channel.

    Parameters
    ----------
    X       : (N, C, T)
    y       : (N,)   class indices 0-based
    channel : channel index to inspect
    """
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=figsize)
    freqs_all = None

    for cls_idx in range(N_CLASSES):
        mask   = y == cls_idx
        if mask.sum() == 0:
            continue
        trials = X[mask, channel, :]              # (n_trials, T)
        psds   = []
        for trial in trials:
            f, p = welch(trial, fs=sfreq, nperseg=min(256, len(trial)))
            psds.append(p)
        freqs_all = f
        mean_psd  = np.mean(psds, axis=0)

        band_mask = (f >= fmin) & (f <= fmax)
        ax.semilogy(f[band_mask], mean_psd[band_mask],
                    label=CLASS_NAMES[cls_idx],
                    color=PALETTE[cls_idx], linewidth=1.8)

    # Mark mu (8-13 Hz) and beta (13-30 Hz) bands
    ax.axvspan(8,  13, alpha=0.08, color="blue",  label="μ band (8-13 Hz)")
    ax.axvspan(13, 30, alpha=0.08, color="green", label="β band (13-30 Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax.set_title(f"PSD per Class — Channel {CH_NAMES[channel]}")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_erp(
    X:         np.ndarray,
    y:         np.ndarray,
    channels:  list[int] = None,
    sfreq:     float     = SFREQ,
    figsize:   tuple     = (16, 8),
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Event-Related Potential: mean signal per class for selected channels.

    Parameters
    ----------
    X        : (N, C, T)
    y        : (N,)   class indices
    channels : list of channel indices to plot (default: C3, Cz, C4 = 10, 7, 13)
    """
    if channels is None:
        channels = [10, 7, 13]   # C3, Cz, C4

    times = np.arange(X.shape[2]) / sfreq
    nrows = len(channels)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, ch_idx in zip(axes, channels):
        for cls_idx in range(N_CLASSES):
            mask = y == cls_idx
            if mask.sum() == 0:
                continue
            mean = X[mask, ch_idx, :].mean(axis=0)
            sem  = X[mask, ch_idx, :].std(axis=0) / np.sqrt(mask.sum())
            ax.plot(times, mean, label=CLASS_NAMES[cls_idx],
                    color=PALETTE[cls_idx], linewidth=1.6)
            ax.fill_between(times, mean - sem, mean + sem,
                            color=PALETTE[cls_idx], alpha=0.15)
        ax.set_ylabel(f"{CH_NAMES[ch_idx]} (µV)")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("Event-Related Potential (mean ± SEM per class)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# MNE-based topographic plots
# ---------------------------------------------------------------------------

def plot_topomap_class(
    X:         np.ndarray,
    y:         np.ndarray,
    band:      tuple[float, float] = (8, 13),   # mu band
    sfreq:     float               = SFREQ,
    figsize:   tuple               = (20, 4),
    save_path: Path | None         = None,
) -> plt.Figure:
    """
    Plot average power topomap per class in a given frequency band.

    Parameters
    ----------
    X    : (N, C, T)
    y    : (N,)
    band : (low_hz, high_hz) frequency band for power extraction
    """
    from scipy.signal import welch

    info    = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    n_classes = N_CLASSES
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    fig.suptitle(f"Band Power Topomap [{band[0]}–{band[1]} Hz]",
                 fontsize=13, fontweight="bold")

    vmin = vmax = None
    all_powers = []

    for cls_idx in range(n_classes):
        mask = y == cls_idx
        if mask.sum() == 0:
            all_powers.append(np.zeros(len(CH_NAMES)))
            continue
        trials = X[mask]   # (n, C, T)
        ch_powers = []
        for ch in range(trials.shape[1]):
            psds = []
            for trial in trials[:, ch, :]:
                f, p = welch(trial, fs=sfreq, nperseg=min(256, len(trial)))
                band_mask = (f >= band[0]) & (f <= band[1])
                psds.append(p[band_mask].mean())
            ch_powers.append(np.mean(psds))
        all_powers.append(np.array(ch_powers))

    all_powers = np.array(all_powers)     # (N_CLASSES, C)
    vmin, vmax = all_powers.min(), all_powers.max()

    for cls_idx, (ax, powers) in enumerate(zip(axes, all_powers)):
        mne.viz.plot_topomap(
            powers, info, axes=ax, show=False,
            vlim=(vmin, vmax), cmap="RdBu_r",
            contours=4,
        )
        ax.set_title(CLASS_NAMES[cls_idx], fontsize=10)

    plt.colorbar(
        plt.cm.ScalarMappable(
            norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap   = "RdBu_r",
        ),
        ax=axes, shrink=0.7, label="Mean power (µV²/Hz)",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Training history plots
# ---------------------------------------------------------------------------

def plot_training_history(
    history:   dict,
    model_name: str        = "Model",
    figsize:   tuple       = FIGSIZE_MD,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : Keras history.history dict
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["loss"],     label="Train",      color="steelblue")
    ax1.plot(epochs, history["val_loss"], label="Validation", color="coral",
             linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    # Accuracy
    acc_key = "accuracy" if "accuracy" in history else "acc"
    ax2.plot(epochs, history[acc_key],          label="Train",      color="steelblue")
    ax2.plot(epochs, history[f"val_{acc_key}"], label="Validation", color="coral",
             linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"{model_name} — Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    model_name:  str       = "Model",
    normalize:   bool      = True,
    figsize:     tuple     = (10, 8),
    save_path:   Path | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix (raw counts or row-normalised).

    Parameters
    ----------
    y_true, y_pred : (N,) integer arrays  (class indices)
    normalize      : if True, show percentages per true class
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, label = ".1%", "Recall per class"
    else:
        cm_display, fmt, label = cm, "d", "Count"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot     = True,
        fmt       = fmt,
        cmap      = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        linewidths  = 0.5,
        ax          = ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} — Confusion Matrix ({label})", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_curves(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    model_name:  str       = "Model",
    figsize:     tuple     = FIGSIZE_MD,
    save_path:   Path | None = None,
) -> plt.Figure:
    """
    One-vs-Rest ROC curves + AUC for all classes.

    Parameters
    ----------
    y_true : (N,)   integer class indices
    y_prob : (N, K) softmax probabilities
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_cls   = y_prob.shape[1]
    y_bin   = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], linewidth=1.6,
                label=f"{class_names[i]}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_precision_recall(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    model_name:  str       = "Model",
    figsize:     tuple     = FIGSIZE_MD,
    save_path:   Path | None = None,
) -> plt.Figure:
    """Precision-Recall curves for all classes."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    n_cls = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_cls):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap           = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=PALETTE[i], linewidth=1.6,
                label=f"{class_names[i]}  (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results:   dict[str, dict],
    metrics:   list[str] = ("accuracy", "f1_macro", "cohen_kappa"),
    figsize:   tuple     = (14, 5),
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Bar-chart comparison across models for selected metrics.

    Parameters
    ----------
    results : {model_name: {metric: value, ...}}
    metrics : list of metric keys to plot
    """
    models = list(results.keys())
    n_met  = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=figsize)
    if n_met == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals   = [results[m].get(metric, 0) for m in models]
        colors = [f"C{i}" for i in range(len(models))]
        bars   = ax.bar(models, vals, color=colors, edgecolor="k", linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_attention_weights(
    weights:    np.ndarray,
    sfreq:      float      = SFREQ,
    n_samples:  int        = 5,
    class_name: str        = "",
    figsize:    tuple      = FIGSIZE_MD,
    save_path:  Path | None = None,
) -> plt.Figure:
    """
    Visualise Bi-LSTM temporal attention weights.

    Parameters
    ----------
    weights : (N, T) attention weight arrays
    n_samples : how many random samples to overlay
    """
    T   = weights.shape[1]
    times = np.arange(T) / sfreq

    fig, ax = plt.subplots(figsize=figsize)
    mean_w = weights.mean(axis=0)
    std_w  = weights.std(axis=0)

    # Sample individual traces
    idx = np.random.choice(len(weights), min(n_samples, len(weights)), replace=False)
    for i in idx:
        ax.plot(times, weights[i], alpha=0.3, linewidth=0.8, color="steelblue")

    ax.plot(times, mean_w, color="steelblue", linewidth=2, label="Mean attention")
    ax.fill_between(times, mean_w - std_w, mean_w + std_w,
                    color="steelblue", alpha=0.2, label="±1 SD")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Attention weight")
    ax.set_title(
        f"Temporal Attention Weights — {class_name}" if class_name else "Temporal Attention Weights",
        fontweight="bold",
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
