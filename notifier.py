"""
notifier.py
Sends email alerts to a caregiver using yagmail (Gmail OAuth / App Password).
"""

import threading
import time

try:
    import yagmail
    YAGMAIL_AVAILABLE = True
except ImportError:
    YAGMAIL_AVAILABLE = False


class Notifier:
    def __init__(
        self,
        sender_email: str = "",
        sender_password: str = "",
        recipient_email: str = "",
        cooldown_seconds: int = 60,
    ):
        """
        sender_email: Gmail address used to send alerts.
        sender_password: Gmail App Password (NOT your regular password).
        recipient_email: Caregiver email address.
        cooldown_seconds: Minimum gap between two emails of the same type.
        """
        self.sender = sender_email
        self.password = sender_password
        self.recipient = recipient_email
        self.cooldown = cooldown_seconds
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def send_alert(self, subject: str, body: str):
        """
        Send an email alert. Respects per-subject cooldown to avoid spam.
        Falls back to console print if yagmail is not installed or not configured.
        """
        now = time.time()
        with self._lock:
            last = self._last_sent.get(subject, 0)
            if now - last < self.cooldown:
                print(f"[Notifier] Cooldown active — skipping '{subject}'")
                return
            self._last_sent[subject] = now

        # Run in a daemon thread so it never blocks the main pipeline
        threading.Thread(
            target=self._send, args=(subject, body), daemon=True
        ).start()

    def send_inactivity_alert(self, duration_seconds: float):
        self.send_alert(
            subject="⚠ Elderly Monitor – Inactivity Alert",
            body=(
                f"No movement has been detected for {duration_seconds:.0f} seconds.\n\n"
                "Please check on the monitored person immediately.\n\n"
                "— Elderly Monitoring System"
            ),
        )

    def send_fall_alert(self):
        self.send_alert(
            subject="🚨 Elderly Monitor – Fall Detected",
            body=(
                "A possible fall has been detected by the monitoring system.\n\n"
                "Please check on the monitored person immediately and call "
                "emergency services if required.\n\n"
                "— Elderly Monitoring System"
            ),
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _send(self, subject: str, body: str):
        if not YAGMAIL_AVAILABLE:
            print(f"[Email – not sent, yagmail missing]\nTo: {self.recipient}\n{subject}\n{body}")
            return

        if not self.sender or not self.recipient:
            print(f"[Email – not configured]\n{subject}\n{body}")
            return

        try:
            yag = yagmail.SMTP(self.sender, self.password)
            yag.send(to=self.recipient, subject=subject, contents=body)
            print(f"[Notifier] Email sent: {subject}")
        except Exception as e:
            print(f"[Notifier] Email failed: {e}")
