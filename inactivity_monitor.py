"""
inactivity_monitor.py
Tracks how long the subject has been inactive and fires a callback when the
threshold is exceeded.
"""

import time
import threading


class InactivityMonitor:
    def __init__(self, threshold_seconds=30, on_alert=None):
        """
        threshold_seconds: seconds of no motion before an alert is fired.
        on_alert: callable(duration_seconds) invoked when threshold is crossed.
        """
        self.threshold = threshold_seconds
        self.on_alert = on_alert
        self._last_motion_time = time.time()
        self._alerted = False          # prevent repeated alerts for same episode
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update_motion(self):
        """Call this every time motion is detected."""
        with self._lock:
            self._last_motion_time = time.time()
            self._alerted = False      # reset so next inactivity period can alert

    def start(self):
        """Start the background checker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_inactive_duration(self):
        """Return seconds since last detected motion."""
        with self._lock:
            return time.time() - self._last_motion_time

    def is_inactive(self):
        return self.get_inactive_duration() >= self.threshold

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _check_loop(self):
        while self._running:
            duration = self.get_inactive_duration()
            if duration >= self.threshold:
                with self._lock:
                    already_alerted = self._alerted
                if not already_alerted:
                    with self._lock:
                        self._alerted = True
                    if callable(self.on_alert):
                        self.on_alert(duration)
            time.sleep(2)   # check every 2 seconds
