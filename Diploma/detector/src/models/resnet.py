"""
=============================================================================
ResNet1D — 1-D Residual Network for EEG Time-Series Classification
=============================================================================
Adapts the ResNet50 residual learning paradigm to 1-D temporal EEG data.

Key design choices vs. ResNet50 (image)
-----------------------------------------
• Conv2D  →  Conv1D           (1-D temporal signal)
• MaxPool stride 2            (temporal downsampling)
• Bottleneck residual blocks  (1×1 → K×1 → 1×1, mirroring ResNet50)
• Global Average Pooling      (replaces flattening, fewer params)
• Kernel size 3               (local temporal patterns)

Architecture layers
-------------------
stem       : Conv1D(64, 7, stride=2) → BN → ELU → MaxPool(3, stride=2)
stage 1    : 3 × BottleneckBlock(64,  64,  256)              [no stride]
stage 2    : 4 × BottleneckBlock(256, 128, 512,  stride=2)
stage 3    : 6 × BottleneckBlock(512, 256, 1024, stride=2)
stage 4    : 3 × BottleneckBlock(1024,512, 2048, stride=2)
head       : GlobalAveragePooling1D → Dense(n_classes, softmax)

Input  : (batch, T, C)  =  (batch, 500, 16)
Output : (batch, n_classes)
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
    RESNET_BASE_FILTERS, RESNET_KERNEL_SIZE,
    DROPOUT_RATE, WEIGHT_DECAY, LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _bn_act(x: tf.Tensor, name: str) -> tf.Tensor:
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("elu", name=f"{name}_act")(x)
    return x


def bottleneck_block(
    x:          tf.Tensor,
    filters_in: int,
    filters:    int,
    filters_out: int,
    stride:     int  = 1,
    l2:         float = WEIGHT_DECAY,
    name:       str   = "block",
) -> tf.Tensor:
    """
    ResNet-50 bottleneck block adapted to 1-D convolutions.

    Bottleneck: Conv1D(1) → Conv1D(K) → Conv1D(1)
    Skip path : identity OR projection Conv1D(1) when dims change.

    Parameters
    ----------
    x            : input tensor  (batch, T, filters_in)
    filters_in   : channels of the input tensor
    filters      : bottleneck width (narrow middle)
    filters_out  : output channels
    stride       : stride for the 3×1 conv (2 = downsample)
    """
    reg  = regularizers.l2(l2)
    skip = x

    # --- Main path ---
    # 1×1 reduction
    x = layers.Conv1D(filters, 1, use_bias=False, kernel_regularizer=reg,
                      name=f"{name}_c1")(x)
    x = _bn_act(x, f"{name}_p1")

    # K×1 temporal conv (with optional stride)
    x = layers.Conv1D(filters, RESNET_KERNEL_SIZE, stride, padding="same",
                      use_bias=False, kernel_regularizer=reg, name=f"{name}_c2")(x)
    x = _bn_act(x, f"{name}_p2")

    # 1×1 expansion
    x = layers.Conv1D(filters_out, 1, use_bias=False, kernel_regularizer=reg,
                      name=f"{name}_c3")(x)
    x = layers.BatchNormalization(name=f"{name}_bn_out")(x)

    # --- Skip / projection path ---
    if stride != 1 or filters_in != filters_out:
        skip = layers.Conv1D(filters_out, 1, stride, use_bias=False,
                             kernel_regularizer=reg, name=f"{name}_proj")(skip)
        skip = layers.BatchNormalization(name=f"{name}_proj_bn")(skip)

    x = layers.Add(name=f"{name}_add")([x, skip])
    x = layers.Activation("elu", name=f"{name}_relu")(x)
    return x


def _make_stage(
    x:          tf.Tensor,
    filters:    int,
    filters_out: int,
    n_blocks:   int,
    stride:     int = 2,
    l2:         float = WEIGHT_DECAY,
    stage:      int = 1,
) -> tf.Tensor:
    """Stack n_blocks bottleneck blocks; first block may stride."""
    filters_in = int(x.shape[-1])
    x = bottleneck_block(x, filters_in, filters, filters_out,
                         stride=stride, l2=l2, name=f"s{stage}_b0")
    for i in range(1, n_blocks):
        x = bottleneck_block(x, filters_out, filters, filters_out,
                             stride=1, l2=l2, name=f"s{stage}_b{i}")
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_resnet1d(
    n_channels:   int   = N_CHANNELS,
    n_samples:    int   = N_SAMPLES,
    n_classes:    int   = N_CLASSES,
    base_filters: int   = RESNET_BASE_FILTERS,
    dropout_rate: float = DROPOUT_RATE,
    l2:           float = WEIGHT_DECAY,
) -> keras.Model:
    """
    Build ResNet1D model.

    Input  : (batch, n_samples, n_channels) = (batch, 500, 16)
    Output : (batch, n_classes)
    """
    inp = keras.Input(shape=(n_samples, n_channels), name="eeg_input")

    # -----------------------------------------------------------------------
    # Stem
    # -----------------------------------------------------------------------
    x = layers.Conv1D(base_filters, 7, strides=2, padding="same",
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("elu", name="stem_act")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same",
                            name="stem_pool")(x)
    # shape: (batch, ~125, 64)

    # -----------------------------------------------------------------------
    # Residual stages  (mirroring ResNet-50 block counts)
    # -----------------------------------------------------------------------
    # Stage 1 — 3 blocks, no downsampling
    x = _make_stage(x, filters=64,  filters_out=256,  n_blocks=3,
                    stride=1, l2=l2, stage=1)
    # Stage 2 — 4 blocks, downsample ×2
    x = _make_stage(x, filters=128, filters_out=512,  n_blocks=4,
                    stride=2, l2=l2, stage=2)
    # Stage 3 — 6 blocks, downsample ×2
    x = _make_stage(x, filters=256, filters_out=1024, n_blocks=6,
                    stride=2, l2=l2, stage=3)
    # Stage 4 — 3 blocks, downsample ×2
    x = _make_stage(x, filters=512, filters_out=2048, n_blocks=3,
                    stride=2, l2=l2, stage=4)

    # -----------------------------------------------------------------------
    # Classification head
    # -----------------------------------------------------------------------
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)
    x = layers.Dense(
        n_classes,
        activation  = "softmax",
        name        = "output",
        kernel_regularizer = regularizers.l2(l2),
    )(x)

    model = keras.Model(inputs=inp, outputs=x, name="ResNet1D")
    return model


# ---------------------------------------------------------------------------
# Lightweight variant (fewer params for quick experiments)
# ---------------------------------------------------------------------------

def build_resnet1d_lite(
    n_channels:   int   = N_CHANNELS,
    n_samples:    int   = N_SAMPLES,
    n_classes:    int   = N_CLASSES,
    base_filters: int   = 32,
    dropout_rate: float = DROPOUT_RATE,
    l2:           float = WEIGHT_DECAY,
) -> keras.Model:
    """
    Lightweight ResNet1D with 3 stages and fewer filters.
    Suitable for limited GPU memory or quick prototyping.
    """
    inp = keras.Input(shape=(n_samples, n_channels), name="eeg_input")

    x = layers.Conv1D(base_filters, 7, strides=2, padding="same",
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("elu", name="stem_act")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same",
                            name="stem_pool")(x)

    x = _make_stage(x, filters=32,  filters_out=128, n_blocks=2,
                    stride=1, l2=l2, stage=1)
    x = _make_stage(x, filters=64,  filters_out=256, n_blocks=3,
                    stride=2, l2=l2, stage=2)
    x = _make_stage(x, filters=128, filters_out=512, n_blocks=2,
                    stride=2, l2=l2, stage=3)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)
    x = layers.Dense(
        n_classes, activation="softmax",
        name="output",
        kernel_regularizer=regularizers.l2(l2),
    )(x)

    model = keras.Model(inputs=inp, outputs=x, name="ResNet1D_Lite")
    return model


# ---------------------------------------------------------------------------
# Compiled model factory
# ---------------------------------------------------------------------------

def get_compiled_resnet(
    lite:          bool  = False,
    learning_rate: float = LEARNING_RATE,
    **kwargs,
) -> keras.Model:
    builder = build_resnet1d_lite if lite else build_resnet1d
    model   = builder(**kwargs)
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
    model = build_resnet1d_lite()
    model.summary(line_length=80)
    dummy = tf.zeros((4, N_SAMPLES, N_CHANNELS))
    out   = model(dummy, training=False)
    print(f"\nOutput shape : {out.shape}")
    print(f"Total params : {model.count_params():,}")
