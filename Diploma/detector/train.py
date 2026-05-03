"""
=============================================================================
EEG Motor Classification — Training Script
=============================================================================
Usage
-----
    python train.py --model eegnet --epochs 80 --batch 64
    python train.py --model resnet --lite
    python train.py --model bilstm --variant deep
    python train.py --model all    # trains all three sequentially

Features
--------
  • Cross-subject split (48 train / 6 val / 6 test)
  • Class-weight balancing for the imbalanced label distribution
  • Training-time data augmentation
  • Early stopping + ReduceLROnPlateau + model checkpoint
  • Full metrics suite saved to results/
  • TensorBoard logging
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, RANDOM_SEED,
    N_CLASSES, CKPT_DIR, RESULTS_DIR,
)
from src.data.loader      import load_cross_subject_splits, compute_class_weights
from src.data.preprocessing import (
    preprocess, to_eegnet_format, to_sequence_format, augment,
)
from src.models.eegnet import get_compiled_eegnet
from src.models.resnet import get_compiled_resnet
from src.models.bilstm import get_compiled_bilstm
from src.utils.metrics import evaluate, per_class_metrics, build_comparison_table

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data pipeline helpers
# ---------------------------------------------------------------------------

def prepare_data(
    splits:     dict,
    model_type: str,
    augment_train: bool = True,
) -> tuple[dict, dict]:
    """
    Preprocess and reshape data for a given model type.

    Parameters
    ----------
    splits     : output of load_cross_subject_splits()
    model_type : 'eegnet' | 'resnet' | 'bilstm'
    augment_train : apply data augmentation to the training set

    Returns
    -------
    data   : dict {split: (X, y)}
    shapes : dict with input_shape info
    """
    data = {}
    for split_name, (X_raw, y, _) in splits.items():
        # Preprocessing
        X = preprocess(X_raw, do_filter=True, do_car=True, do_zscore=True)

        # Augment training set only
        if split_name == "train" and augment_train:
            X, y = augment(X, y)

        # Reshape for target model
        if model_type == "eegnet":
            X = to_eegnet_format(X)   # (N, 1, 16, 500)
        else:
            X = to_sequence_format(X) # (N, 500, 16)

        data[split_name] = (X, y)

    shapes = {"input_shape": data["train"][0].shape[1:]}
    return data, shapes


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(model_name: str) -> list:
    ckpt_path = CKPT_DIR / f"{model_name}_best.keras"
    log_dir   = RESULTS_DIR / "logs" / model_name

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = str(ckpt_path),
            monitor           = "val_accuracy",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = PATIENCE // 2,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir          = str(log_dir),
            histogram_freq   = 0,
            write_graph      = False,
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULTS_DIR / f"{model_name}_log.csv"),
            separator = ",",
            append    = False,
        ),
    ]


# ---------------------------------------------------------------------------
# Single model training loop
# ---------------------------------------------------------------------------

def train_model(
    model_type:    str,
    splits:        dict,
    epochs:        int   = EPOCHS,
    batch_size:    int   = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    augment:       bool  = True,
    # model-specific
    lite:          bool  = False,
    bilstm_variant: str  = "standard",
) -> dict:
    """
    Train a single model and return evaluation metrics on the test set.

    Returns
    -------
    dict with keys: model_name, test_metrics, per_class_df, history
    """
    log.info(f"{'='*60}")
    log.info(f"  Training: {model_type.upper()}")
    log.info(f"{'='*60}")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    data, shapes = prepare_data(splits, model_type, augment_train=augment)
    (X_train, y_train) = data["train"]
    (X_val,   y_val)   = data["val"]
    (X_test,  y_test)  = data["test"]

    # Class weights for imbalanced dataset
    raw_y_train = splits["train"][1]
    cw = compute_class_weights(raw_y_train)
    log.info(f"Class weights: {cw}")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    if model_type == "eegnet":
        model_name = "EEGNet"
        model = get_compiled_eegnet(learning_rate=learning_rate)

    elif model_type == "resnet":
        model_name = "ResNet1D" + ("_Lite" if lite else "")
        model = get_compiled_resnet(lite=lite, learning_rate=learning_rate)

    elif model_type == "bilstm":
        model_name = f"BiLSTM_{bilstm_variant}"
        model = get_compiled_bilstm(variant=bilstm_variant,
                                    learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.summary(line_length=80)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    callbacks = build_callbacks(model_name)
    t0 = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        class_weight    = cw,
        callbacks       = callbacks,
        verbose         = 1,
    )
    elapsed = time.time() - t0
    log.info(f"Training complete in {elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------
    log.info("Evaluating on test set ...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    test_metrics = evaluate(y_true=y_test, y_pred=y_pred,
                            y_prob=y_prob, verbose=True)
    pc_df = per_class_metrics(y_test, y_pred)

    # Save results
    _save_results(model_name, test_metrics, pc_df, history.history)

    return {
        "model_name":  model_name,
        "test_metrics": test_metrics,
        "per_class":   pc_df,
        "history":     history.history,
        "model":       model,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_results(
    name:    str,
    metrics: dict,
    pc_df,
    history: dict,
) -> None:
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pc_df.to_csv(out / "per_class.csv")

    hist_df_path = out / "history.json"
    with open(hist_df_path, "w") as f:
        # Convert numpy float32 → native float for JSON
        serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(serialisable, f, indent=2)

    log.info(f"Results saved to {out}/")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EEG motor classification models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default="eegnet",
                   choices=["eegnet", "resnet", "bilstm", "all"],
                   help="Model to train")
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",      type=float, default=LEARNING_RATE)
    p.add_argument("--no-aug",  action="store_true",
                   help="Disable data augmentation")
    # ResNet flags
    p.add_argument("--lite",    action="store_true",
                   help="[ResNet] use lite variant")
    # BiLSTM flags
    p.add_argument("--variant", default="standard",
                   choices=["standard", "deep", "lite"],
                   help="[BiLSTM] model variant")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    log.info("Loading dataset splits ...")
    splits = load_cross_subject_splits()

    models_to_train = (
        ["eegnet", "resnet", "bilstm"] if args.model == "all" else [args.model]
    )

    all_results: dict[str, dict] = {}
    for mt in models_to_train:
        result = train_model(
            model_type    = mt,
            splits        = splits,
            epochs        = args.epochs,
            batch_size    = args.batch,
            learning_rate = args.lr,
            augment       = not args.no_aug,
            lite          = args.lite,
            bilstm_variant= args.variant,
        )
        all_results[result["model_name"]] = result["test_metrics"]

    if len(all_results) > 1:
        table = build_comparison_table(all_results)
        print("\n" + "="*60)
        print("  MODEL COMPARISON")
        print("="*60)
        print(table.to_string())
        table.to_csv(RESULTS_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
