"""
motion_detection.py
Detects motion using frame differencing with OpenCV.
"""

import cv2
import numpy as np
import threading
import time


class MotionDetector:
    def __init__(self, sensitivity=25, min_area=500):
        """
        sensitivity: pixel difference threshold (lower = more sensitive)
        min_area: minimum contour area to count as motion
        """
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.prev_frame = None
        self.motion_detected = False
        self.last_motion_time = time.time()
        self._lock = threading.Lock()

    def detect(self, frame):
        """
        Analyse a single frame. Returns (annotated_frame, motion_bool).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return frame, False

        # Frame differencing
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion = False
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                motion = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update rolling previous frame (blend for stability)
        self.prev_frame = gray

        with self._lock:
            self.motion_detected = motion
            if motion:
                self.last_motion_time = time.time()

        # Overlay status text
        label = "MOTION DETECTED" if motion else "No Motion"
        color = (0, 0, 255) if motion else (0, 200, 0)
        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        return frame, motion

    def get_last_motion_time(self):
        with self._lock:
            return self.last_motion_time

    def is_motion(self):
        with self._lock:
            return self.motion_detected
