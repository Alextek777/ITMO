"""
=============================================================================
EEG Motor Imagery Classification — Global Configuration
=============================================================================
Dataset : MILimbEEG (Mendeley, v2)
Task    : 8-class motor action recognition (cross-subject)
Authors : ITMO Diploma project
=============================================================================
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = Path("/home/kalex/projects/ITMO/Diploma/datasets/MILimbEEG/MILimbEEG")
RESULTS_DIR  = ROOT_DIR / "results"
CKPT_DIR     = ROOT_DIR / "checkpoints"

RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# EEG Signal Parameters
# ---------------------------------------------------------------------------
SFREQ        = 125          # Hz  (OpenBCI Cyton+Daisy)
N_CHANNELS   = 16
N_SAMPLES    = 500          # samples per trial  →  4.0 s
DURATION_SEC = N_SAMPLES / SFREQ   # 4.0 s

# 10-10 channel names (columns 0-15 in CSV → channels 1-16 in the map image)
CH_NAMES = [
    "FC5", "F3",  "Fz",  "F4",
    "FC6", "FC3", "FC4", "Cz",
    "T7",  "CP5", "C3",  "CP3",
    "CP4", "C4",  "CP6", "T8",
]
CH_TYPES = ["eeg"] * N_CHANNELS

# Hardware bandpass already applied: 5-50 Hz + notch at 60 Hz
# We apply a soft bandpass in software for MNE compatibility
BANDPASS_LOW  = 1.0    # Hz  (re-reference + drift removal)
BANDPASS_HIGH = 49.0   # Hz
NOTCH_FREQ    = 50.0   # Hz  (EU grid; dataset uses 60 Hz notch by hardware)

# ---------------------------------------------------------------------------
# Dataset / Class Labels
# ---------------------------------------------------------------------------
LABEL_MAP = {
    1: "BEO",   # Baseline Eyes Open
    2: "CLH",   # Closing Left Hand
    3: "CRH",   # Closing Right Hand
    4: "DLH",   # Dorsal Flexion Left Foot
    5: "PLF",   # Plantar Flexion Left Foot
    6: "DRF",   # Dorsal Flexion Right Foot
    7: "PRF",   # Plantar Flexion Right Foot
    8: "Rest",  # Rest / no action
}

N_CLASSES      = len(LABEL_MAP)          # 8
CLASS_NAMES    = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]

# Map original labels (1-8) → indices (0-7)
LABEL_TO_IDX   = {k: k - 1 for k in LABEL_MAP}
IDX_TO_LABEL   = {v: k for k, v in LABEL_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Cross-Subject Split  (subjects 1-60)
# ---------------------------------------------------------------------------
ALL_SUBJECTS   = list(range(1, 61))      # S1 … S60
TRAIN_SUBJECTS = list(range(1,  49))     # S1–S48   (80%)
VAL_SUBJECTS   = list(range(49, 55))     # S49–S54  (10%)
TEST_SUBJECTS  = list(range(55, 61))     # S55–S60  (10%)

# ---------------------------------------------------------------------------
# Training Hyper-parameters (shared defaults)
# ---------------------------------------------------------------------------
BATCH_SIZE     = 64
EPOCHS         = 80
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4   # L2 regularisation (keras kernel_regularizer)
DROPOUT_RATE   = 0.5
PATIENCE       = 15     # early-stopping patience

# ---------------------------------------------------------------------------
# EEGNet-specific
# ---------------------------------------------------------------------------
EEGNET_F1          = 8      # number of temporal filters
EEGNET_D           = 2      # depth multiplier (depthwise)
EEGNET_F2          = 16     # EEGNET_F1 * EEGNET_D
EEGNET_KERNEL_SIZE = 64     # temporal kernel ≈ 0.5 s at 125 Hz

# ---------------------------------------------------------------------------
# ResNet1D-specific
# ---------------------------------------------------------------------------
RESNET_BASE_FILTERS  = 64
RESNET_KERNEL_SIZE   = 3

# ---------------------------------------------------------------------------
# Bi-LSTM-specific
# ---------------------------------------------------------------------------
BILSTM_UNITS     = [128, 64]   # units per layer
BILSTM_DENSE     = 64

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
