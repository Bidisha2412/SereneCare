"""
video_pipeline.py — CareWatch Video → Joint Coordinates
=========================================================
Converts raw video files to joint-coordinate DataFrames using MediaPipe Pose,
then feeds them through the same feature engineering pipeline.

Architecture:
  Video file
    → cv2 frame reader (frame by frame — no full video in RAM)
    → MediaPipe Pose → 33 landmark (x, y, z, visibility)
    → landmark DataFrame (one row per frame)
    → label assignment from filename or sidecar .txt
    → features.process_chunk()

Landmark mapping to CareWatch skeleton:
  MediaPipe uses 33 landmarks. We map them to a CareWatch-compatible naming
  convention so the same feature engineering code handles both tabular and
  video-derived data.
"""

import os
import logging
from typing import Generator, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    VIDEO_EXTENSIONS, VIDEO_FPS_TARGET, MEDIAPIPE_MODEL_COMPLEXITY,
    WINDOW_SIZE, STRIDE, CLASS_LABELS
)
from src.ml.features import process_chunk

logger = logging.getLogger("carewatch.video")

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark index → CareWatch joint name mapping
# ─────────────────────────────────────────────────────────────────────────────
MP_TO_CAREWATCH: Dict[int, str] = {
    0:  "nose",
    7:  "left_ear",
    8:  "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",            # ≈ left_upper_leg
    24: "right_hip",           # ≈ right_upper_leg
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot",
    32: "right_foot",
}

# Derived joints not directly in MediaPipe (computed as midpoints)
DERIVED_JOINTS = {
    "pelvis":    (23, 24),     # midpoint of hips
    "neck":      (11, 12),     # midpoint of shoulders
    "head":      (0,  0),      # nose approximation
}


# ─────────────────────────────────────────────────────────────────────────────
# Label extraction from file name / sidecar
# ─────────────────────────────────────────────────────────────────────────────

def _label_from_filename(path: str) -> int:
    """
    Try to infer class label from the video file name.
    Expects patterns like: fall_001.mp4, normal_subject3.mp4, etc.
    """
    base = os.path.splitext(os.path.basename(path))[0].lower()
    for name, idx in CLASS_LABELS.items():
        if name in base:
            return idx
    # Check sidecar label file (same name, .txt extension with one word)
    txt_path = os.path.splitext(path)[0] + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            content = f.read().strip().lower()
            return CLASS_LABELS.get(content, -1)
    return -1  # unknown


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe pose initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_pose_model():
    """Lazily import and create MediaPipe Pose. Raises if not installed."""
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode       = False,
            model_complexity        = MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks        = True,
            enable_segmentation     = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )
        return pose
    except ImportError:
        raise ImportError(
            "MediaPipe not installed. Run: pip install mediapipe"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Frame → landmark row
# ─────────────────────────────────────────────────────────────────────────────

def _frame_to_landmark_row(
    results,
    img_w: int,
    img_h: int,
    frame_idx: int,
    label_int: int,
) -> Optional[dict]:
    """
    Convert MediaPipe Pose results for one frame into a flat dict
    compatible with CareWatch joint naming.
    Returns None if no pose detected.
    """
    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks.landmark
    row = {"frame": frame_idx, "label_int": label_int}

    # Direct landmarks
    for mp_idx, joint_name in MP_TO_CAREWATCH.items():
        lm = lms[mp_idx]
        # Scale to pixel coordinates (x=horizontal, y=vertical from top)
        row[f"{joint_name}_x"] = lm.x * img_w
        row[f"{joint_name}_y"] = lm.y * img_h
        row[f"{joint_name}_z"] = lm.z * img_w  # depth (optional)
        row[f"{joint_name}_vis"] = lm.visibility

    # Derived: pelvis = mid of hips
    for derived, (idx_a, idx_b) in DERIVED_JOINTS.items():
        a, b = lms[idx_a], lms[idx_b]
        row[f"{derived}_x"] = ((a.x + b.x) / 2) * img_w
        row[f"{derived}_y"] = ((a.y + b.y) / 2) * img_h

    # Approximate l5 / l3 / t12 / t8 by interpolating pelvis → neck
    pelvis_x = row["pelvis_x"]; pelvis_y = row["pelvis_y"]
    neck_x   = row["neck_x"];   neck_y   = row["neck_y"]
    for name, frac in [("l5", 0.2), ("l3", 0.4), ("t12", 0.55), ("t8", 0.7)]:
        row[f"{name}_x"] = pelvis_x + frac * (neck_x - pelvis_x)
        row[f"{name}_y"] = pelvis_y + frac * (neck_y - pelvis_y)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Video file → landmark DataFrame (streaming, no full video in RAM)
# ─────────────────────────────────────────────────────────────────────────────

def video_to_landmarks(
    video_path: str,
    target_fps: float = VIDEO_FPS_TARGET,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrames of landmark rows in chunks of ~WINDOW_SIZE*10 frames
    to keep memory bounded.

    Uses frame-skipping to resample to target_fps.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not installed. Run: pip install opencv-python")

    label_int = _label_from_filename(video_path)
    pose      = _get_pose_model()
    cap       = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip      = max(1, round(src_fps / target_fps))  # frame skip interval

    CHUNK_FRAMES = WINDOW_SIZE * 10   # yield every 300 frames (10 windows)
    buffer       = []
    frame_idx    = 0
    kept_idx     = 0

    logger.info(f"Processing video: {video_path} | FPS={src_fps:.1f} | skip={skip}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        row = _frame_to_landmark_row(results, img_w, img_h, kept_idx, label_int)
        if row:
            buffer.append(row)
        kept_idx += 1

        # Yield buffer in chunks
        if len(buffer) >= CHUNK_FRAMES:
            yield pd.DataFrame(buffer)
            buffer = buffer[-WINDOW_SIZE:]  # keep last window for continuity

    cap.release()
    pose.close()

    if buffer:
        yield pd.DataFrame(buffer)


# ─────────────────────────────────────────────────────────────────────────────
# High-level: video → feature DataFrame generator
# ─────────────────────────────────────────────────────────────────────────────

def stream_video_chunks(
    data_dir: str,
    window: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Generator[pd.DataFrame, None, None]:
    """
    Walk data_dir for video files, extract landmarks, compute features.
    Yields feature DataFrames (same schema as tabular pipeline output).
    """
    from src.ml.ingestion import discover_files
    video_files = discover_files(data_dir)["video"]
    if not video_files:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    for vpath in sorted(video_files):
        for landmark_chunk in video_to_landmarks(vpath):
            feat_df = process_chunk(landmark_chunk, window=window, stride=stride)
            if not feat_df.empty:
                yield feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Real-time inference helper (single frame)
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeVideoPipeline:
    """
    Maintains a rolling frame buffer; call update(frame) each frame.
    When buffer has enough frames, extract features for inference.

    Usage:
        pipe = RealTimeVideoPipeline()
        for frame in camera:
            feat = pipe.update(frame)
            if feat is not None:
                probs = model.predict(feat)
    """

    def __init__(self, window: int = WINDOW_SIZE, stride: int = STRIDE):
        self._window  = window
        self._stride  = stride
        self._buffer  = []
        self._frame_n = 0
        self._pose    = _get_pose_model()

    def update(self, bgr_frame: np.ndarray) -> Optional[pd.DataFrame]:
        """
        Feed one BGR frame. Returns feature DataFrame when buffer is full,
        None otherwise.
        """
        h, w = bgr_frame.shape[:2]
        try:
            import cv2
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        except ImportError:
            rgb = bgr_frame[:, :, ::-1]

        results = self._pose.process(rgb)
        row = _frame_to_landmark_row(results, w, h, self._frame_n, -1)
        self._frame_n += 1

        if row:
            self._buffer.append(row)

        if len(self._buffer) >= self._window:
            df      = pd.DataFrame(self._buffer[-self._window:])
            feat_df = process_chunk(df, window=self._window, stride=self._window)
            # Slide buffer
            self._buffer = self._buffer[-self._stride:]
            return feat_df if not feat_df.empty else None

        return None

    def close(self):
        self._pose.close()
