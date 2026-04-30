"""
features.py — CareWatch Feature Engineering
=============================================
Converts raw joint-coordinate frames into a rich feature vector designed
specifically to discriminate between:
  0 = Normal movement
  1 = Fall
  2 = Heart attack risk
  3 = Panic attack

Design rationale
────────────────
• Falls       → sudden high jerk, posture collapse (bounding box goes wide),
                COM height drops sharply, impact spike in acceleration
• Heart attack → movement SLOWS then near-freeze, strong left-right asymmetry,
                unusual spine/torso lean, tremor in extremities
• Panic        → rapid erratic motion, high tremor frequency, irregular
                variance bursts, elevated but inconsistent velocity

Pipeline
────────
  raw_chunk (DataFrame)
    → _impute_and_clean
    → _derive_kinematics         (velocity / accel / jerk per joint)
    → _derive_posture_features   (spine angle, COM height, bounding box)
    → _derive_symmetry           (left-right asymmetry)
    → _compute_window_features   (sliding-window statistical descriptors)
    → feature vector (numpy / DataFrame row per window)
"""

import re
import warnings
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch          # for tremor frequency analysis
import warnings; warnings.filterwarnings("ignore")

from config import (
    JOINT_COORD_PATTERN, WINDOW_SIZE, STRIDE,
    STAT_DESCRIPTORS, DERIVED_SIGNALS
)

logger = logging.getLogger("carewatch.features")

# ─────────────────────────────────────────────────────────────────────────────
# Constants — adjust to match YOUR skeleton topology
# ─────────────────────────────────────────────────────────────────────────────

# Spine joints used for posture angle (vertical alignment)
SPINE_JOINTS  = ["pelvis", "l5", "l3", "t12", "t8", "neck"]

# Left-right limb pairs for symmetry score
SYMMETRY_PAIRS = [
    ("left_upper_leg",  "right_upper_leg"),
    ("left_lower_leg",  "right_lower_leg"),
    ("left_foot",       "right_foot"),
]

# Joints to use for centre-of-mass approximation
COM_JOINTS    = ["pelvis", "l5", "l3", "t12", "t8", "neck", "head"]

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Imputation & Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _impute_and_clean(df: pd.DataFrame, joint_cols: List[str]) -> pd.DataFrame:
    """
    Forward-fill → backward-fill → zero-fill joint coordinates.
    Clip extreme outliers (>5 IQR from median) per column.
    """
    df = df.copy()

    # Clip outliers — per column, >5× IQR from median
    for col in joint_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 5 * iqr, q3 + 5 * iqr
        df[col] = df[col].clip(lo, hi)

    # Forward then backward fill temporal gaps
    df[joint_cols] = df[joint_cols].ffill().bfill().fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Kinematics (velocity / acceleration / jerk)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_kinematics(df: pd.DataFrame, joint_cols: List[str]) -> pd.DataFrame:
    """
    Compute per-joint velocity, acceleration, jerk using finite differences.
    Assumes rows are ordered by time (frame number).

    Returns df with added columns:  {joint}_{x|y}_{vel|accel|jerk}
    """
    df = df.copy()

    for col in joint_cols:
        vel   = df[col].diff().fillna(0.0)                   # Δpos / frame
        accel = vel.diff().fillna(0.0)                        # Δvel / frame
        jerk  = accel.diff().fillna(0.0)                      # Δaccel / frame
        df[f"{col}_vel"]   = vel
        df[f"{col}_accel"] = accel
        df[f"{col}_jerk"]  = jerk

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Posture Features
# ─────────────────────────────────────────────────────────────────────────────

def _get_joint_xy(df: pd.DataFrame, joint_name: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Return (x_col, y_col) series for a given joint name, or (None, None)."""
    x_col = f"{joint_name}_x"
    y_col = f"{joint_name}_y"
    x = df[x_col] if x_col in df.columns else None
    y = df[y_col] if y_col in df.columns else None
    return x, y


def _derive_posture_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns:
      spine_angle         — angle of trunk from vertical (degrees)
      com_height          — y-coordinate of centre of mass proxy
      bbox_aspect_ratio   — body bounding-box width/height (→ fall marker)
      head_pelvis_dist    — total body height proxy
    """
    df = df.copy()

    # ── Spine angle ──────────────────────────────────────────────────────
    # Fit line through spine joint y-coordinates against their x-positions
    spine_x_cols = [f"{j}_x" for j in SPINE_JOINTS if f"{j}_x" in df.columns]
    spine_y_cols = [f"{j}_y" for j in SPINE_JOINTS if f"{j}_y" in df.columns]

    if len(spine_x_cols) >= 2:
        xs = df[spine_x_cols].values   # shape (N, k)
        ys = df[spine_y_cols].values

        # Vectorised slope across spine joints per frame
        x_mean = xs.mean(axis=1, keepdims=True)
        y_mean = ys.mean(axis=1, keepdims=True)
        num    = ((xs - x_mean) * (ys - y_mean)).sum(axis=1)
        denom  = ((xs - x_mean) ** 2).sum(axis=1) + EPS
        slope  = num / denom
        df["spine_angle"] = np.degrees(np.arctan(np.abs(slope)))
    else:
        df["spine_angle"] = 0.0

    # ── Centre-of-mass height proxy ───────────────────────────────────────
    com_y_cols = [f"{j}_y" for j in COM_JOINTS if f"{j}_y" in df.columns]
    if com_y_cols:
        df["com_height"] = df[com_y_cols].mean(axis=1)
    else:
        df["com_height"] = 0.0

    # ── Bounding-box aspect ratio ─────────────────────────────────────────
    # Use ALL _x and ALL _y columns
    x_cols  = [c for c in df.columns if c.endswith("_x") and "vel" not in c and "accel" not in c and "jerk" not in c]
    y_cols  = [c for c in df.columns if c.endswith("_y") and "vel" not in c and "accel" not in c and "jerk" not in c]

    if x_cols and y_cols:
        x_range = df[x_cols].max(axis=1) - df[x_cols].min(axis=1)
        y_range = df[y_cols].max(axis=1) - df[y_cols].min(axis=1) + EPS
        df["bbox_aspect_ratio"] = x_range / y_range
    else:
        df["bbox_aspect_ratio"] = 0.0

    # ── Head-to-pelvis distance ───────────────────────────────────────────
    head_x, head_y = _get_joint_xy(df, "head")
    pelvis_x, pelvis_y = _get_joint_xy(df, "pelvis")
    if head_y is not None and pelvis_y is not None:
        df["head_pelvis_dist"] = (head_y - pelvis_y).abs()
    else:
        df["head_pelvis_dist"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Left-Right Symmetry
# ─────────────────────────────────────────────────────────────────────────────

def _derive_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean distance between left and right counterpart joints.
    Asymmetry spike → one-sided motor control loss (heart attack signal).
    """
    df = df.copy()
    for (left, right) in SYMMETRY_PAIRS:
        lx, ly = _get_joint_xy(df, left)
        rx, ry = _get_joint_xy(df, right)
        if lx is not None and rx is not None:
            asym = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
            col  = f"asym_{left.replace('left_', '')}"
            df[col] = asym

    # Aggregate: mean + std of all asymmetry signals
    asym_cols = [c for c in df.columns if c.startswith("asym_")]
    if asym_cols:
        df["mean_asymmetry"] = df[asym_cols].mean(axis=1)
        df["std_asymmetry"]  = df[asym_cols].std(axis=1).fillna(0)
    else:
        df["mean_asymmetry"] = 0.0
        df["std_asymmetry"]  = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Tremor Index (spectral)
# ─────────────────────────────────────────────────────────────────────────────

def _tremor_index_window(velocity_window: np.ndarray, fs: float = 30.0) -> float:
    """
    Power in tremor band (3–10 Hz) as fraction of total signal power.
    High tremor → panic attack or Parkinson-like motion.
    Requires at least 8 samples (nperseg constraint).
    """
    n = len(velocity_window)
    if n < 8:
        return 0.0
    nperseg = min(n, 256)
    try:
        freqs, psd = welch(velocity_window, fs=fs, nperseg=nperseg)
        tremor_band = (freqs >= 3) & (freqs <= 10)
        total_power = psd.sum() + EPS
        tremor_power = psd[tremor_band].sum()
        return float(tremor_power / total_power)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Sliding Window Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _window_stats(arr: np.ndarray, prefix: str) -> dict:
    """Compute statistical descriptors for a 1-D array window."""
    if len(arr) == 0:
        return {f"{prefix}_{d}": 0.0 for d in STAT_DESCRIPTORS}
    return {
        f"{prefix}_mean":  float(np.mean(arr)),
        f"{prefix}_std":   float(np.std(arr)),
        f"{prefix}_min":   float(np.min(arr)),
        f"{prefix}_max":   float(np.max(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_skew":  float(skew(arr)) if len(arr) > 2 else 0.0,
        f"{prefix}_kurt":  float(kurtosis(arr)) if len(arr) > 3 else 0.0,
    }


def compute_window_features(
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    stride: int = STRIDE,
    label_col: str = "label_int"
) -> pd.DataFrame:
    """
    Slide a window of `window` frames with `stride` step.
    For each window, extract:
      • Per-signal statistical descriptors (mean/std/min/max/range/skew/kurt)
      • Composite physics features (jerk spike, COM drop, bbox ratio spike)
      • Tremor index from pelvis velocity (spectral)
      • Label = majority vote over window frames

    Returns a DataFrame where each row = one window's feature vector.
    This is the output fed to the ML model.
    """
    n       = len(df)
    records = []

    # Identify the velocity columns to use for tremor
    vel_cols = [c for c in df.columns if c.endswith("_vel")]

    for start in range(0, n - window + 1, stride):
        end   = start + window
        win   = df.iloc[start:end]
        rec   = {}

        # ── Statistical descriptors for all numeric signals ───────────────
        numeric_cols = win.select_dtypes(include=[np.number]).columns
        # Exclude the label columns
        sig_cols = [c for c in numeric_cols if c not in ("label_int",)]

        for col in sig_cols:
            arr = win[col].values.astype(float)
            rec.update(_window_stats(arr, col))

        # ── Composite physics features ────────────────────────────────────

        # 1. Global jerk spike — max absolute jerk across all joints
        jerk_cols = [c for c in win.columns if c.endswith("_jerk")]
        if jerk_cols:
            rec["global_jerk_spike"] = float(win[jerk_cols].abs().max(axis=None))
            rec["mean_jerk"]         = float(win[jerk_cols].abs().mean(axis=None))

        # 2. COM height drop — difference from window start to end
        if "com_height" in win.columns:
            rec["com_height_drop"]  = float(win["com_height"].iloc[0] - win["com_height"].iloc[-1])
            rec["com_height_delta"] = float(win["com_height"].max() - win["com_height"].min())

        # 3. Bounding-box aspect ratio change (narrow → wide = fall)
        if "bbox_aspect_ratio" in win.columns:
            rec["bbox_ratio_end"]    = float(win["bbox_aspect_ratio"].iloc[-1])
            rec["bbox_ratio_spike"]  = float(win["bbox_aspect_ratio"].max() - win["bbox_aspect_ratio"].min())

        # 4. Spine angle change
        if "spine_angle" in win.columns:
            rec["spine_angle_max"]   = float(win["spine_angle"].max())
            rec["spine_angle_delta"] = float(win["spine_angle"].max() - win["spine_angle"].min())

        # 5. Tremor index — computed on pelvis_x_vel or first vel column
        if vel_cols:
            chosen_vel = "pelvis_x_vel" if "pelvis_x_vel" in vel_cols else vel_cols[0]
            rec["tremor_index"] = _tremor_index_window(win[chosen_vel].values)

        # 6. Velocity entropy — erratic motion indicator
        if vel_cols:
            all_vels = win[vel_cols].values.flatten()
            all_vels = np.abs(all_vels[~np.isnan(all_vels)])
            if all_vels.sum() > 0:
                p = all_vels / (all_vels.sum() + EPS)
                p = p[p > 0]
                rec["velocity_entropy"] = float(-np.sum(p * np.log(p + EPS)))
            else:
                rec["velocity_entropy"] = 0.0

        # 7. Freeze ratio — fraction of frames with near-zero velocity
        if vel_cols:
            mean_abs_vel = win[vel_cols].abs().mean(axis=1)
            rec["freeze_ratio"] = float((mean_abs_vel < 0.5).mean())

        # ── Label: majority vote ──────────────────────────────────────────
        if label_col in win.columns:
            valid = win[label_col][win[label_col] >= 0]
            rec["label_int"] = int(valid.mode().iloc[0]) if len(valid) else -1
        else:
            rec["label_int"] = -1

        records.append(rec)

    if not records:
        logger.warning("No windows generated — chunk may be shorter than window size")
        return pd.DataFrame()

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level: process one raw chunk end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def process_chunk(
    df: pd.DataFrame,
    window: int  = WINDOW_SIZE,
    stride: int  = STRIDE,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for one chunk.
    Input : raw DataFrame chunk (from ingestion.py)
    Output: windowed feature DataFrame ready for model training.
    """
    from src.ml.ingestion import detect_joint_columns  # avoid circular at module level

    joint_cols = detect_joint_columns(df)
    if not joint_cols:
        logger.warning("No joint coordinate columns detected in chunk — skipping")
        return pd.DataFrame()

    df = _impute_and_clean(df, joint_cols)
    df = _derive_kinematics(df, joint_cols)
    df = _derive_posture_features(df)
    df = _derive_symmetry(df)

    feat_df = compute_window_features(df, window=window, stride=stride)
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Utility: discover feature column names (needed by training & inference)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(sample_df: pd.DataFrame) -> List[str]:
    """Return all columns except labels and index columns."""
    exclude = {"label_int", "label", "Unnamed: 0", "index"}
    return [c for c in sample_df.columns if c not in exclude]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke-test with synthetic data
    np.random.seed(0)
    N = 200
    joint_names = ["pelvis", "l5", "l3", "t12", "t8", "neck",
                   "left_upper_leg", "right_upper_leg",
                   "left_lower_leg", "right_lower_leg",
                   "left_foot", "right_foot"]
    cols = {}
    for j in joint_names:
        cols[f"{j}_x"] = np.random.randn(N) * 100
        cols[f"{j}_y"] = np.random.randn(N) * 100
    cols["label_int"] = np.random.randint(0, 4, N)
    df = pd.DataFrame(cols)

    feat = process_chunk(df)
    print(f"Input rows:     {N}")
    print(f"Output windows: {len(feat)}")
    print(f"Feature dims:   {len(get_feature_columns(feat))}")
    print(feat.head(2))
