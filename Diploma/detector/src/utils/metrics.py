"""
=============================================================================
EEG Motor Classification — Metrics & Evaluation Utilities
=============================================================================
Computes a comprehensive suite of metrics for multi-class classification:

  • Accuracy (overall + per-class)
  • Macro / Weighted F1-Score
  • Cohen's Kappa (chance-corrected agreement)
  • Matthews Correlation Coefficient (MCC)
  • One-vs-Rest AUC (macro / weighted)
  • Balanced Accuracy

All functions accept NumPy arrays and return plain Python scalars or dicts,
making them framework-agnostic and easy to log to any backend.
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CLASS_NAMES, N_CLASSES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_prob:    Optional[np.ndarray] = None,
    verbose:   bool = True,
) -> dict[str, float]:
    """
    Compute the full evaluation suite.

    Parameters
    ----------
    y_true : (N,)     integer ground-truth class indices
    y_pred : (N,)     integer predicted class indices
    y_prob : (N, K)   softmax probabilities (optional; needed for AUC)
    verbose : print a formatted report

    Returns
    -------
    metrics : dict mapping metric name → float value
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"]         = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"]= balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"]         = f1_score(y_true, y_pred, average="macro",
                                           zero_division=0)
    metrics["f1_weighted"]      = f1_score(y_true, y_pred, average="weighted",
                                           zero_division=0)
    metrics["cohen_kappa"]      = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"]              = matthews_corrcoef(y_true, y_pred)

    if y_prob is not None:
        n_cls  = y_prob.shape[1]
        y_bin  = label_binarize(y_true, classes=list(range(n_cls)))
        try:
            metrics["auc_macro"]    = roc_auc_score(y_bin, y_prob,
                                                    average="macro",
                                                    multi_class="ovr")
            metrics["auc_weighted"] = roc_auc_score(y_bin, y_prob,
                                                    average="weighted",
                                                    multi_class="ovr")
        except ValueError as e:
            log.warning(f"AUC computation failed: {e}")

    if verbose:
        _print_report(y_true, y_pred, metrics)

    return metrics


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """
    Return per-class precision, recall, F1 and support as a DataFrame.

    Parameters
    ----------
    y_true, y_pred : (N,) integer arrays

    Returns
    -------
    pd.DataFrame  shape (K, 4)  columns: precision recall f1 support
    """
    report = classification_report(
        y_true, y_pred,
        labels       = list(range(len(class_names))),
        target_names = class_names,
        output_dict  = True,
        zero_division= 0,
    )
    rows = {name: report[name] for name in class_names if name in report}
    df   = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    df.index.name = "class"
    return df


# ---------------------------------------------------------------------------
# Summary table across multiple models
# ---------------------------------------------------------------------------

def build_comparison_table(
    results: dict[str, dict[str, float]],
    metrics: list[str] = ("accuracy", "f1_macro", "cohen_kappa",
                          "auc_macro", "mcc"),
) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of model results.

    Parameters
    ----------
    results : {model_name: metrics_dict}
    metrics : which metrics to include

    Returns
    -------
    pd.DataFrame  shape (n_models, n_metrics)
    """
    rows = {}
    for model_name, model_metrics in results.items():
        rows[model_name] = {m: model_metrics.get(m, float("nan")) for m in metrics}
    df = pd.DataFrame(rows).T
    df.index.name   = "Model"
    df.columns.name = "Metric"
    return df.round(4)


# ---------------------------------------------------------------------------
# Formatted console report
# ---------------------------------------------------------------------------

def _print_report(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    metrics: dict[str, float],
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  EVALUATION REPORT")
    print(sep)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  F1  (macro)       : {metrics['f1_macro']:.4f}")
    print(f"  F1  (weighted)    : {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's κ         : {metrics['cohen_kappa']:.4f}")
    print(f"  MCC               : {metrics['mcc']:.4f}")
    if "auc_macro" in metrics:
        print(f"  AUC (macro OvR)   : {metrics['auc_macro']:.4f}")
    print(sep)

    report = classification_report(
        y_true, y_pred,
        labels       = list(range(N_CLASSES)),
        target_names = CLASS_NAMES,
        zero_division= 0,
    )
    print("\nPer-class report:\n")
    print(report)


# ---------------------------------------------------------------------------
# Keras callback — logs metrics to dict at end of training
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Lightweight container that stores evaluation metrics across epochs.
    Pass to a Keras custom callback or call manually after each epoch.
    """

    def __init__(self) -> None:
        self.history: list[dict] = []

    def log(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        epoch:  int = 0,
    ) -> dict:
        m = evaluate(y_true, y_pred, y_prob, verbose=False)
        m["epoch"] = epoch
        self.history.append(m)
        return m

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index("epoch")

    def best(self, key: str = "f1_macro") -> dict:
        return max(self.history, key=lambda d: d.get(key, 0.0))
