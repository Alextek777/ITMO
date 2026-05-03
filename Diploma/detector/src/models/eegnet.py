"""
=============================================================================
EEGNet — Compact Convolutional Neural Network for EEG
=============================================================================
Reference
---------
Lawhern V.J. et al. (2018). "EEGNet: a compact convolutional neural network
for EEG-based brain–computer interfaces."
Journal of Neural Engineering, 15(5), 056013.
https://doi.org/10.1088/1741-2552/aace8c

Input format (channels_last — CPU-compatible)
---------------------------------------------
  (batch, n_channels, n_samples, 1)
   batch=N, height=C=16, width=T=500, channels=1

Use to_eegnet_format(X) to reshape from (N, C, T) → (N, C, T, 1).

Architecture Overview
---------------------
Input  : (batch, C, T, 1)   C=16 channels, T=500 samples

Block 1 — Temporal + Depthwise Spatial convolution
  Conv2D(F1, (1, kernel_size))       →  temporal filter bank
  BatchNormalization
  DepthwiseConv2D((C, 1), D)         →  spatial filter (one per electrode)
  BatchNormalization
  Activation('elu')
  AveragePooling2D((1, 4))
  Dropout(dropout)

Block 2 — Separable convolution (temporal smoothing)
  SeparableConv2D(F2, (1, 16))
  BatchNormalization
  Activation('elu')
  AveragePooling2D((1, 8))
  Dropout(dropout)

Classifier
  Flatten
  Dense(n_classes, softmax)

Parameter count ≈ 2 400 (original paper, 2-class)
For 8-class with defaults: F1=8, D=2, F2=16 → ~4 200 params
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
    EEGNET_F1, EEGNET_D, EEGNET_F2, EEGNET_KERNEL_SIZE,
    DROPOUT_RATE, WEIGHT_DECAY, LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_eegnet(
    n_channels:   int   = N_CHANNELS,
    n_samples:    int   = N_SAMPLES,
    n_classes:    int   = N_CLASSES,
    F1:           int   = EEGNET_F1,
    D:            int   = EEGNET_D,
    F2:           int   = EEGNET_F2,
    kernel_size:  int   = EEGNET_KERNEL_SIZE,
    dropout_rate: float = DROPOUT_RATE,
    l2:           float = WEIGHT_DECAY,
    norm_rate:    float = 0.25,      # max_norm constraint on DepthwiseConv
) -> keras.Model:
    """
    Build and return the EEGNet model.

    Input shape : (batch, 1, n_channels, n_samples)
    Output shape: (batch, n_classes)

    Parameters
    ----------
    n_channels   : number of EEG electrodes (16)
    n_samples    : number of time samples (500)
    n_classes    : number of output classes (8)
    F1           : number of temporal filters (8)
    D            : depth multiplier for depthwise conv (2)
    F2           : number of separable conv filters = F1*D (16)
    kernel_size  : temporal kernel length in samples (64 ≈ 0.5 s)
    dropout_rate : dropout probability (0.5)
    l2           : L2 regularisation coefficient
    norm_rate    : max-norm for DepthwiseConv kernel constraint
    """
    reg = regularizers.l2(l2)
    # Input layout (channels_last — CPU-compatible):
    #   (batch, n_channels, n_samples, 1)
    #   height = n_channels (EEG electrodes)
    #   width  = n_samples  (time)
    #   depth  = 1          (single "colour" channel)
    # Use to_eegnet_format() from preprocessing.py to produce this shape.

    inp = keras.Input(shape=(n_channels, n_samples, 1), name="eeg_input")

    # -----------------------------------------------------------------------
    # Block 1: Temporal Filter Bank + Depthwise Spatial Filter
    # -----------------------------------------------------------------------
    # Temporal convolution: 1-row × kernel_size-column kernel scans over time
    x = layers.Conv2D(
        filters            = F1,
        kernel_size        = (1, kernel_size),
        padding            = "same",
        use_bias           = False,
        kernel_regularizer = reg,
        name               = "temporal_conv",
    )(inp)
    # Shape: (batch, n_channels, n_samples, F1)
    x = layers.BatchNormalization(name="bn1")(x)

    # Depthwise conv: (n_channels, 1) kernel covers all electrodes at once
    x = layers.DepthwiseConv2D(
        kernel_size           = (n_channels, 1),
        depth_multiplier      = D,
        use_bias              = False,
        depthwise_regularizer = reg,
        depthwise_constraint  = keras.constraints.MaxNorm(norm_rate, axis=[0, 1, 2]),
        name                  = "depthwise_conv",
    )(x)
    # Shape: (batch, 1, n_samples, F1*D)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("elu", name="act1")(x)
    x = layers.AveragePooling2D(pool_size=(1, 4), name="pool1")(x)
    # Shape: (batch, 1, n_samples//4, F1*D)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # -----------------------------------------------------------------------
    # Block 2: Separable Convolution (pointwise temporal smoothing)
    # -----------------------------------------------------------------------
    x = layers.SeparableConv2D(
        filters               = F2,
        kernel_size           = (1, 16),
        padding               = "same",
        use_bias              = False,
        depthwise_regularizer = reg,
        pointwise_regularizer = reg,
        name                  = "separable_conv",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("elu", name="act2")(x)
    x = layers.AveragePooling2D(pool_size=(1, 8), name="pool2")(x)
    # Shape: (batch, 1, n_samples//32, F2)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # -----------------------------------------------------------------------
    # Classifier
    # -----------------------------------------------------------------------
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        n_classes,
        kernel_constraint = keras.constraints.MaxNorm(0.25),
        activation        = "softmax",
        name              = "output",
    )(x)

    model = keras.Model(inputs=inp, outputs=x, name="EEGNet")
    return model


# ---------------------------------------------------------------------------
# Compiled model factory
# ---------------------------------------------------------------------------

def get_compiled_eegnet(
    learning_rate: float = LEARNING_RATE,
    **kwargs,
) -> keras.Model:
    """Return a compiled EEGNet model ready for training."""
    model = build_eegnet(**kwargs)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "sparse_categorical_crossentropy",
        metrics   = [
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_eegnet()
    model.summary(line_length=80)
    # Input: (batch, n_channels, n_samples, 1)
    dummy = tf.zeros((4, N_CHANNELS, N_SAMPLES, 1))
    out   = model(dummy, training=False)
    print(f"\nOutput shape : {out.shape}")
    print(f"Total params : {model.count_params():,}")
