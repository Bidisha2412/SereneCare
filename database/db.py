"""
db.py
SQLite database helper for logging alerts and events.
"""

import sqlite3
import threading
from datetime import datetime

DB_PATH = "elderly_monitor.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    type      TEXT    NOT NULL DEFAULT 'general',
                    message   TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
            """)
            conn.commit()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def insert_alert(self, message: str, alert_type: str = "general"):
        """Insert a new alert record. Returns the new row id."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO alerts (type, message, timestamp) VALUES (?, ?, ?)",
                    (alert_type, message, ts),
                )
                conn.commit()
                return cur.lastrowid

    def fetch_alerts(self, limit: int = 50):
        """Return the most recent `limit` alerts as a list of dicts."""
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(row) for row in rows]

    def count_alerts(self):
        with self._lock:
            with self._connect() as conn:
                return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def clear_alerts(self):
        """Delete all records (useful for testing / demo reset)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM alerts")
                conn.commit()
