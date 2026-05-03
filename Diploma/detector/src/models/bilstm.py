"""
=============================================================================
Bi-LSTM — Bidirectional LSTM for EEG Sequence Classification
=============================================================================
Motivation
----------
EEG signals have rich temporal dependencies in both forward and backward
directions.  A Bidirectional LSTM captures context from past AND future
time steps at each position, making it well-suited for motor imagery where
frequency modulations evolve across the full 4-second epoch.

Architecture Overview
---------------------
Input      : (batch, T, C)  =  (batch, 500, 16)

Projection : Dense(64)         linear embedding of raw electrode values
             ↓
Layer 1    : Bidirectional(LSTM(128, return_sequences=True))
             LayerNormalization
             Dropout(0.3)
             ↓
Layer 2    : Bidirectional(LSTM(64, return_sequences=True))
             LayerNormalization
             Dropout(0.3)
             ↓
Attention  : (optional) temporal self-attention aggregation
             OR GlobalAveragePooling1D
             ↓
Dense      : Dense(64, elu) → Dropout(0.3)
             ↓
Output     : Dense(n_classes, softmax)

Variants
--------
• build_bilstm        : standard stack with attention
• build_bilstm_deep   : 3 Bi-LSTM layers, more capacity
• build_bilstm_lite   : single Bi-LSTM layer, minimal params
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    N_CHANNELS, N_SAMPLES, N_CLASSES,
    BILSTM_UNITS, BILSTM_DENSE,
    DROPOUT_RATE, WEIGHT_DECAY, LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Temporal Self-Attention Layer (Bahdanau-style)
# ---------------------------------------------------------------------------

class TemporalAttention(layers.Layer):
    """
    Soft temporal attention that produces a context vector as a weighted
    sum over the time dimension.

    Input  : (batch, T, H)
    Output : (batch, H)
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.W       = layers.Dense(units, use_bias=True)
        self.v       = layers.Dense(1, use_bias=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        # x: (batch, T, H)
        score   = self.v(tf.nn.tanh(self.W(x)))      # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)        # (batch, T, 1)
        context = tf.reduce_sum(weights * x, axis=1)  # (batch, H)
        return context, tf.squeeze(weights, axis=-1)  # context + attention map

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


# ---------------------------------------------------------------------------
# Standard Bi-LSTM with attention
# ---------------------------------------------------------------------------

def build_bilstm(
    n_channels:    int   = N_CHANNELS,
    n_samples:     int   = N_SAMPLES,
    n_classes:     int   = N_CLASSES,
    lstm_units:    list  = None,
    dense_units:   int   = BILSTM_DENSE,
    dropout_rate:  float = DROPOUT_RATE,
    l2:            float = WEIGHT_DECAY,
    use_attention: bool  = True,
    embed_dim:     int   = 64,
) -> keras.Model:
    """
    Build Bi-LSTM model with optional temporal attention.

    Input  : (batch, n_samples, n_channels) = (batch, 500, 16)
    Output : (batch, n_classes)

    Parameters
    ----------
    lstm_units    : list of hidden units per Bi-LSTM layer
    dense_units   : units in the penultimate Dense layer
    dropout_rate  : dropout probability
    l2            : L2 regularisation
    use_attention : use soft temporal attention instead of GlobalAvgPool
    embed_dim     : linear projection dimension for input channels
    """
    if lstm_units is None:
        lstm_units = BILSTM_UNITS  # [128, 64]

    reg = regularizers.l2(l2)
    inp = keras.Input(shape=(n_samples, n_channels), name="eeg_input")

    # -----------------------------------------------------------------------
    # Input projection  (lift raw 16-dim EEG to embed_dim)
    # -----------------------------------------------------------------------
    x = layers.Dense(embed_dim, kernel_regularizer=reg, name="proj")(inp)
    x = layers.LayerNormalization(name="proj_ln")(x)

    # -----------------------------------------------------------------------
    # Bi-LSTM stack
    # -----------------------------------------------------------------------
    for i, units in enumerate(lstm_units):
        return_seq = True   # all layers return sequences; attention handles last
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences = return_seq,
                dropout          = 0.1,
                recurrent_dropout= 0.0,   # keep 0 for GPU efficiency
                kernel_regularizer = reg,
            ),
            name = f"bilstm_{i}",
        )(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{i}")(x)

    # -----------------------------------------------------------------------
    # Temporal aggregation
    # -----------------------------------------------------------------------
    if use_attention:
        attn_layer = TemporalAttention(units=lstm_units[-1] * 2, name="attention")
        x, _ = attn_layer(x)
    else:
        x = layers.GlobalAveragePooling1D(name="gap")(x)

    # -----------------------------------------------------------------------
    # Classification head
    # -----------------------------------------------------------------------
    x = layers.Dense(dense_units, activation="elu",
                     kernel_regularizer=reg, name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)
    x = layers.Dense(n_classes, activation="softmax",
                     kernel_regularizer=reg, name="output")(x)

    model = keras.Model(inputs=inp, outputs=x, name="BiLSTM")
    return model


# ---------------------------------------------------------------------------
# Deeper variant (3 Bi-LSTM layers)
# ---------------------------------------------------------------------------

def build_bilstm_deep(
    n_channels:   int   = N_CHANNELS,
    n_samples:    int   = N_SAMPLES,
    n_classes:    int   = N_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    l2:           float = WEIGHT_DECAY,
) -> keras.Model:
    """3-layer Bi-LSTM with residual connections between layers."""
    return build_bilstm(
        n_channels   = n_channels,
        n_samples    = n_samples,
        n_classes    = n_classes,
        lstm_units   = [256, 128, 64],
        dense_units  = 128,
        dropout_rate = dropout_rate,
        l2           = l2,
        use_attention= True,
        embed_dim    = 64,
    )


# ---------------------------------------------------------------------------
# Lite variant (single layer, fast iteration)
# ---------------------------------------------------------------------------

def build_bilstm_lite(
    n_channels:   int   = N_CHANNELS,
    n_samples:    int   = N_SAMPLES,
    n_classes:    int   = N_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    l2:           float = WEIGHT_DECAY,
) -> keras.Model:
    """Single Bi-LSTM layer — fast sanity-check / baseline."""
    return build_bilstm(
        n_channels   = n_channels,
        n_samples    = n_samples,
        n_classes    = n_classes,
        lstm_units   = [64],
        dense_units  = 32,
        dropout_rate = dropout_rate,
        l2           = l2,
        use_attention= False,
        embed_dim    = 32,
    )


# ---------------------------------------------------------------------------
# Compiled model factory
# ---------------------------------------------------------------------------

def get_compiled_bilstm(
    variant:       str   = "standard",   # "standard" | "deep" | "lite"
    learning_rate: float = LEARNING_RATE,
    **kwargs,
) -> keras.Model:
    """
    Return a compiled Bi-LSTM model.

    Parameters
    ----------
    variant : one of 'standard', 'deep', 'lite'
    """
    builders = {
        "standard": build_bilstm,
        "deep":     build_bilstm_deep,
        "lite":     build_bilstm_lite,
    }
    if variant not in builders:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(builders)}")

    model = builders[variant](**kwargs)
    model.compile(
        optimizer = keras.optimizers.Adam(
            learning_rate = learning_rate,
            clipnorm      = 1.0,          # gradient clipping for LSTM stability
        ),
        loss    = "sparse_categorical_crossentropy",
        metrics = [
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_bilstm()
    model.summary(line_length=80)
    dummy = tf.zeros((4, N_SAMPLES, N_CHANNELS))
    out   = model(dummy, training=False)
    print(f"\nOutput shape : {out.shape}")
    print(f"Total params : {model.count_params():,}")
