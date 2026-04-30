"""
fall_detection.py
Heuristic fall detection based on bounding-box aspect ratio changes.

How it works:
  A standing person has a tall bounding box (height >> width).
  When someone falls, they become horizontal (width >= height).
  We track the ratio history and flag a sudden change as a potential fall.

Optional: if you install mediapipe you can enable pose-landmark-based
detection by setting USE_POSE_LANDMARKS = True at the top of this file.
"""

import cv2
import numpy as np
import time
import threading

USE_POSE_LANDMARKS = False  # set True if mediapipe is installed

if USE_POSE_LANDMARKS:
    try:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
        _pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
        )
    except ImportError:
        USE_POSE_LANDMARKS = False


class FallDetector:
    def __init__(self, on_fall=None, history_len=10, ratio_threshold=0.85):
        """
        on_fall: callable() invoked when a fall is detected.
        history_len: number of frames to keep in ratio history.
        ratio_threshold: width/height ratio above which person is deemed horizontal.
        """
        self.on_fall = on_fall
        self.ratio_threshold = ratio_threshold
        self.history_len = history_len
        self._ratio_history = []
        self._fall_detected = False
        self._last_fall_time = 0
        self._cooldown = 10          # seconds before another alert
        self._lock = threading.Lock()

        # Background subtractor for isolating the subject
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )

    def analyse(self, frame):
        """
        Process a frame. Returns (annotated_frame, fall_bool).
        """
        fall = False

        if USE_POSE_LANDMARKS:
            fall = self._pose_based(frame)
        else:
            fall = self._heuristic(frame)

        # Cooldown guard
        now = time.time()
        with self._lock:
            if fall and (now - self._last_fall_time) > self._cooldown:
                self._fall_detected = True
                self._last_fall_time = now
                if callable(self.on_fall):
                    self.on_fall()
            else:
                self._fall_detected = False

        if fall:
            cv2.putText(
                frame, "⚠ FALL DETECTED", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
            )

        return frame, fall

    # ------------------------------------------------------------------ #
    # Heuristic approach                                                   #
    # ------------------------------------------------------------------ #

    def _heuristic(self, frame):
        fg_mask = self._bg_sub.apply(frame)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        # Take the largest contour as the person
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 3000:
            return False

        x, y, w, h = cv2.boundingRect(largest)
        if h == 0:
            return False

        ratio = w / h   # horizontal if > 1

        self._ratio_history.append(ratio)
        if len(self._ratio_history) > self.history_len:
            self._ratio_history.pop(0)

        # Detect if current ratio suddenly spiked compared to recent average
        if len(self._ratio_history) >= 5:
            avg_prev = np.mean(self._ratio_history[:-3])
            curr = self._ratio_history[-1]
            if curr >= self.ratio_threshold and avg_prev < 0.7:
                return True

        return False

    # ------------------------------------------------------------------ #
    # Pose-based (optional mediapipe)                                      #
    # ------------------------------------------------------------------ #

    def _pose_based(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _pose.process(rgb)
        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Key landmark indices: nose(0), left_hip(23), right_hip(24)
        nose = landmarks[0]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        nose_y = nose.y * h
        hip_y = ((left_hip.y + right_hip.y) / 2) * h

        # If nose is at roughly the same height as hips → fallen
        vertical_diff = abs(nose_y - hip_y)
        if vertical_diff < h * 0.15:
            return True

        return False

    def is_fallen(self):
        with self._lock:
            return self._fall_detected
