"""
config.py — CareWatch Training Pipeline Configuration
======================================================
Single place to change ALL pipeline behaviour.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("CAREWATCH_DATA_DIR", "./data")   # root of your dataset
OUTPUT_DIR      = os.getenv("CAREWATCH_OUT_DIR",  "./output") # models / scalers / reports
FEATURE_CACHE   = os.path.join(OUTPUT_DIR, "feature_cache")   # parquet chunks cache

# ─────────────────────────────────────────────────────────────────────────────
# DATA FORMAT  ("csv" | "parquet" | "video" | "auto")
# ─────────────────────────────────────────────────────────────────────────────
DATA_FORMAT = "auto"            # auto-detects from file extensions

# ─────────────────────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["normal", "fall", "heart_attack", "panic"]
CLASS_LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES  = len(CLASS_NAMES)

# Column in tabular data that holds the label
LABEL_COLUMN = "label"        # string class name column  OR
LABEL_INT_COLUMN = None       # integer label column (set one or the other)

# ─────────────────────────────────────────────────────────────────────────────
# TABULAR / JOINT DATA
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_ROWS   = 50_000          # rows per CSV chunk (tune to RAM)
PARQUET_ROW_GROUP = 100_000    # row-group size when writing feature cache

# Joint coordinate column pattern — regex matched against column names
# The dataset has columns like pelvis_x, pelvis_y, l5_x, …
JOINT_COORD_PATTERN = r"^(pelvis|l[0-9]+|t[0-9]+|neck|head|.*shoulder.*|.*elbow.*|.*wrist.*|.*hand.*|.*hip.*|.*knee.*|.*ankle.*|.*foot.*|.*toe.*|.*upper.*|.*lower.*|clavicle|sternum)_(x|y|z)$"
INDEX_COLUMN        = "Unnamed: 0"    # drop this if present

# Temporal window for sequence features (frames)
WINDOW_SIZE  = 30       # 1 second at 30 fps
STRIDE       = 15       # 50 % overlap

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PIPELINE (only used when DATA_FORMAT == "video")
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_FPS_TARGET  = 30           # resample to this FPS
VIDEO_EXTENSIONS  = [".mp4", ".avi", ".mov", ".mkv"]
MEDIAPIPE_MODEL_COMPLEXITY = 1   # 0 fast / 1 balanced / 2 accurate

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# Statistical descriptors computed per window
STAT_DESCRIPTORS = ["mean", "std", "min", "max", "range", "skew", "kurt"]

# Physics-based derived signals (computed before windowing)
DERIVED_SIGNALS = [
    "velocity",       # Δposition / Δt per joint
    "acceleration",   # Δvelocity / Δt
    "jerk",           # Δacceleration / Δt  (high jerk → sudden impact)
    "angle",          # joint angles (knee, hip, spine tilt)
    "symmetry",       # left-right limb difference
    "posture_score",  # vertical spine alignment index
    "com_height",     # centre-of-mass height proxy
    "bounding_box",   # body bounding-box aspect ratio (fall signature)
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Primary model: "lgbm" | "xgb" | "sgd" | "rf" | "lstm"
MODEL_TYPE = "lgbm"

# ── LightGBM params (incremental training via file-based dataset) ─────────
LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        NUM_CLASSES,
    "metric":           "multi_logloss",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l2":        1.0,
    "is_unbalance":     True,   # handles class imbalance internally
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}
LGBM_NUM_BOOST_ROUND  = 500
LGBM_EARLY_STOP_ROUNDS = 30

# ── XGBoost params ────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":         "multi:softprob",
    "num_class":         NUM_CLASSES,
    "eval_metric":       "mlogloss",
    "eta":               0.05,
    "max_depth":         6,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "lambda":            1.0,
    "seed":              42,
    "nthread":           -1,
}
XGB_NUM_ROUNDS = 500
XGB_EARLY_STOP = 30

# ── SGDClassifier (true online / incremental) ─────────────────────────────
SGD_PARAMS = {
    "loss":          "modified_huber",   # supports predict_proba
    "penalty":       "l2",
    "alpha":         1e-4,
    "max_iter":      1,                  # 1 pass per partial_fit call
    "tol":           None,
    "random_state":  42,
    "class_weight":  "balanced",
    "n_jobs":        -1,
}

# ── LSTM params ───────────────────────────────────────────────────────────
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 30
LSTM_BATCH    = 256
LSTM_LR       = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_SPLIT   = 0.15   # fraction held out for early-stop / eval
TEST_SPLIT         = 0.10   # final held-out test set
RANDOM_STATE       = 42
SCALER_TYPE        = "standard"  # "standard" | "robust" | "minmax"

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_THRESHOLD_FALL        = 0.60   # confidence to trigger alert
INFERENCE_THRESHOLD_HEART       = 0.65
INFERENCE_THRESHOLD_PANIC       = 0.65
INFERENCE_HISTORY_WINDOW_FRAMES = 30     # rolling buffer for real-time use
