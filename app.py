"""
app.py  —  SereneCare Elderly Monitoring System
=================================================
Routes:
  /             → login page
  /caregiver    → caregiver dashboard
  /patient      → patient dashboard
  /video        → MJPEG live stream
  /alerts       → alert history JSON
  /status       → system status JSON  (now includes health_risk)
  /risk         → latest ML prediction JSON
  /voice        → trigger voice interaction
  /clear        → clear alerts
  /test_alert   → inject test alert
  /send_message → relay caregiver message to patient
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# ── Local modules ────────────────────────────────────────────────────────────
from motion_detection   import MotionDetector
from inactivity_monitor import InactivityMonitor
from fall_detection     import FallDetector
from voice_ai           import VoiceAI
from notifier           import Notifier
from db                 import Database

# ── ML Health Risk module (in same folder) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from health_risk_model import HealthRiskPredictor, MotionFeatureExtractor

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX         = 0       # 0 = default webcam
INACTIVITY_THRESHOLD = 60      # seconds before inactivity alert
SENDER_EMAIL         = ""      # Gmail address (optional)
SENDER_PASSWORD      = ""      # Gmail App Password (optional)
RECIPIENT_EMAIL      = ""      # caregiver email (optional)
RISK_PREDICT_EVERY   = 2       # run ML prediction every N seconds

# ─────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ─────────────────────────────────────────────────────────────────────────────
app      = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "serenecare-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

db                 = Database()
notifier           = Notifier(SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)
voice_ai           = VoiceAI(model_size="tiny")
motion_detector    = MotionDetector(sensitivity=25, min_area=500)
fall_detector      = FallDetector()
inactivity_monitor = InactivityMonitor(threshold_seconds=INACTIVITY_THRESHOLD)

# ── ML components ─────────────────────────────────────────────────────────────
print("[SereneCare] Loading Health Risk ML model…")
health_predictor  = HealthRiskPredictor()
feature_extractor = MotionFeatureExtractor(window=30)
print("[SereneCare] ML model ready.")

# ── Shared state ──────────────────────────────────────────────────────────────
_camera        = None
_camera_lock   = threading.Lock()
_latest_frame  = None
_frame_lock    = threading.Lock()

_system_status = {
    "monitoring":        False,
    "camera_ok":         False,
    "inactive_duration": 0,
    "motion":            False,
    "fall":              False,
    # Health risk fields — updated by ML loop
    "health_risk_label":      0,
    "health_risk_class":      "Normal",
    "health_risk_confidence": 0.0,
    "health_risk_severity":   "none",
    "health_risk_probs":      {},
}

_last_risk_prediction = {}   # full prediction dict, served at /risk


# ─────────────────────────────────────────────────────────────────────────────
# Alert helper
# ─────────────────────────────────────────────────────────────────────────────
def _fire_alert(message: str, alert_type: str = "general"):
    row_id  = db.insert_alert(message, alert_type)
    payload = {
        "id":        row_id,
        "type":      alert_type,
        "message":   message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    socketio.emit("new_alert", payload)
    print(f"[ALERT] [{alert_type}] {message}")


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def _on_inactivity(duration):
    msg = f"No movement detected for {duration:.0f} seconds."
    _fire_alert(msg, "inactivity")
    notifier.send_inactivity_alert(duration)

inactivity_monitor.on_alert = _on_inactivity


def _on_fall():
    _fire_alert("Possible fall detected by the monitoring system.", "fall")
    notifier.send_fall_alert()
    # voice removed intentionally

fall_detector.on_fall = _on_fall


# ─────────────────────────────────────────────────────────────────────────────
# Health Risk alert handler
# ─────────────────────────────────────────────────────────────────────────────
_RISK_ALERT_MESSAGES = {
    1: {   # Heart Attack Risk
        'alert_type': 'heart_attack_risk',
        'caregiver_msg': (
            "⚠️ HEART ATTACK RISK DETECTED — Monitoring has identified abnormal "
            "joint movement patterns consistent with cardiac distress. "
            "Please check on the patient immediately."
        ),
        'patient_hindi': (
            "सावधान! आपकी हरकतों में असामान्यता पाई गई है। "
            "कृपया शांत रहें और देखभालकर्ता को बुलाएं।"
        ),
    },
    2: {   # Panic Attack
        'alert_type': 'panic_attack_risk',
        'caregiver_msg': (
            "⚠️ PANIC ATTACK DETECTED — Rapid erratic movement and elevated "
            "stress indicators detected. Patient may be experiencing anxiety "
            "or panic. Please check in."
        ),
        'patient_hindi': (
            "घबराएं नहीं। गहरी सांस लें। आप सुरक्षित हैं। "
            "देखभालकर्ता को सूचित कर दिया गया है।"
        ),
    },
}


def _handle_risk_prediction(result: dict):
    """
    Called every RISK_PREDICT_EVERY seconds with the latest ML prediction.
    Updates system_status and fires alerts when needed.
    """
    global _last_risk_prediction, _system_status
    _last_risk_prediction = result

    # Update shared status
    _system_status["health_risk_label"]      = result["label"]
    _system_status["health_risk_class"]      = result["class_name"]
    _system_status["health_risk_confidence"] = result["confidence"]
    _system_status["health_risk_severity"]   = result["severity"]
    _system_status["health_risk_probs"]      = result["probabilities"]

    # Push live risk update to all browsers
    socketio.emit("risk_update", {
        "label":       result["label"],
        "class_name":  result["class_name"],
        "confidence":  result["confidence"],
        "severity":    result["severity"],
        "probs":       result["probabilities"],
        "overridden":  result["overridden_to_false_alarm"],
        "timestamp":   time.strftime("%H:%M:%S"),
    })

    # Fire alert only when model says we should (confidence + cooldown)
    if result["should_alert"] and result["label"] in _RISK_ALERT_MESSAGES:
        info = _RISK_ALERT_MESSAGES[result["label"]]
        _fire_alert(info["caregiver_msg"], info["alert_type"])

        # Also send Hindi notification to patient dashboard
        socketio.emit("patient_risk_notification", {
            "class_name":   result["class_name"],
            "hindi_msg":    info["patient_hindi"],
            "confidence":   result["confidence"],
            "severity":     result["severity"],
        })


# ─────────────────────────────────────────────────────────────────────────────
# Background monitoring + ML prediction loop
# ─────────────────────────────────────────────────────────────────────────────
def _monitoring_loop():
    global _camera, _latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    with _camera_lock:
        _camera = cap

    if not cap.isOpened():
        _system_status["camera_ok"] = False
        print("[Monitor] Camera not available — demo mode.")
        _demo_loop()
        return

    _system_status["camera_ok"]  = True
    _system_status["monitoring"] = True
    inactivity_monitor.start()

    last_predict_time = time.time()

    while _system_status["monitoring"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame, motion = motion_detector.detect(frame)
        frame, fall   = fall_detector.analyse(frame)

        if motion:
            inactivity_monitor.update_motion()

        inactive_dur = inactivity_monitor.get_inactive_duration()

        _system_status["motion"]            = motion
        _system_status["fall"]              = fall
        _system_status["inactive_duration"] = inactive_dur

        # Feed every frame into feature extractor
        feature_extractor.update(frame, motion, fall, inactive_dur)

        # Run ML prediction every RISK_PREDICT_EVERY seconds
        now = time.time()
        if now - last_predict_time >= RISK_PREDICT_EVERY:
            last_predict_time = now
            features = feature_extractor.get_features(fall, inactive_dur)
            result   = health_predictor.predict_from_motion(features)
            _handle_risk_prediction(result)

        with _frame_lock:
            _latest_frame = frame.copy()

        time.sleep(0.03)

    cap.release()


def _demo_loop():
    """Placeholder when no camera — ML still runs on synthetic features."""
    _system_status["monitoring"] = True
    inactivity_monitor.start()

    img = np.zeros((480, 640, 3), dtype="uint8")
    img[:] = (25, 35, 48)
    cv2.putText(img, "No Camera — Demo Mode", (110, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 180, 220), 2)
    cv2.putText(img, "ML Risk Detection Active", (120, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (52, 211, 153), 2)

    last_predict = time.time()

    while _system_status["monitoring"]:
        inactive_dur = inactivity_monitor.get_inactive_duration()
        _system_status["inactive_duration"] = inactive_dur

        with _frame_lock:
            _latest_frame = img.copy()

        now = time.time()
        if now - last_predict >= RISK_PREDICT_EVERY:
            last_predict = now
            # Demo: use normal baseline features
            features = {
                'joint_velocity':    0.30,
                'acceleration':      0.20,
                'tremor_index':      0.05,
                'posture_angle':     5.0,
                'movement_variance': 0.10,
                'heart_rate_sim':    72.0,
                'lateral_sway':      2.0,
                'limb_asymmetry':    0.05,
            }
            result = health_predictor.predict_from_motion(features)
            _handle_risk_prediction(result)

        time.sleep(1)


threading.Thread(target=_monitoring_loop, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG stream
# ─────────────────────────────────────────────────────────────────────────────
def _generate_frames():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/caregiver")
def caregiver_dashboard():
    return render_template("caregiver_dashboard.html")


@app.route("/patient")
def patient_dashboard():
    return render_template("patient_dashboard.html")


@app.route("/video")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/alerts")
def get_alerts():
    return jsonify(db.fetch_alerts(limit=50))


@app.route("/status")
def get_status():
    status = dict(_system_status)
    status["alert_count"] = db.count_alerts()
    return jsonify(status)


@app.route("/risk")
def get_risk():
    """Latest ML health-risk prediction as JSON."""
    return jsonify(_last_risk_prediction or {
        "label": 0, "class_name": "Normal",
        "confidence": 0, "severity": "none",
        "probabilities": {}
    })


@app.route("/voice", methods=["POST"])
def trigger_voice():
    def _interact():
        text, response = voice_ai.interact()
        _fire_alert(f'Voice: "{text}" → "{response}"', "voice")
    threading.Thread(target=_interact, daemon=True).start()
    return jsonify({"status": "listening"})


@app.route("/clear", methods=["POST"])
def clear_alerts():
    db.clear_alerts()
    socketio.emit("alerts_cleared", {})
    return jsonify({"status": "cleared"})


@app.route("/test_alert", methods=["POST"])
def test_alert():
    _fire_alert("Test alert triggered from dashboard.", "test")
    return jsonify({"status": "ok"})


@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json() or {}
    socketio.emit("caregiver_message", {
        "message":   data.get("message", ""),
        "sender":    data.get("sender",  "Caregiver"),
        "timestamp": time.strftime("%H:%M"),
    })
    return jsonify({"status": "sent"})


@app.route("/simulate_risk", methods=["POST"])
def simulate_risk():
    """
    Demo-only: inject a synthetic risk scenario.
    Body: { "scenario": "heart_attack" | "panic" | "normal" | "false_alarm" }
    """
    data      = request.get_json() or {}
    scenario  = data.get("scenario", "normal")

    SCENARIOS = {
        "heart_attack": {
            'joint_velocity': 0.04, 'acceleration': 0.65, 'tremor_index': 0.55,
            'posture_angle': 42.0,  'movement_variance': 0.48, 'heart_rate_sim': 118.0,
            'lateral_sway': 13.0,   'limb_asymmetry': 0.75,
        },
        "panic": {
            'joint_velocity': 0.82, 'acceleration': 1.15, 'tremor_index': 0.68,
            'posture_angle': 11.0,  'movement_variance': 0.72, 'heart_rate_sim': 132.0,
            'lateral_sway': 7.5,    'limb_asymmetry': 0.30,
        },
        "normal": {
            'joint_velocity': 0.30, 'acceleration': 0.20, 'tremor_index': 0.05,
            'posture_angle':  5.0,  'movement_variance': 0.10, 'heart_rate_sim': 72.0,
            'lateral_sway':   2.0,  'limb_asymmetry': 0.05,
        },
        "false_alarm": {
            'joint_velocity': 0.95, 'acceleration': 0.90, 'tremor_index': 0.10,
            'posture_angle': 26.0,  'movement_variance': 0.56, 'heart_rate_sim': 92.0,
            'lateral_sway':  5.2,   'limb_asymmetry': 0.18,
        },
    }

    features = SCENARIOS.get(scenario, SCENARIOS["normal"])

    # Temporarily lower cooldown threshold for demo
    health_predictor._cooldown = {}
    result = health_predictor.predict_from_motion(features)
    result["should_alert"] = scenario in ("heart_attack", "panic")
    _handle_risk_prediction(result)

    if result["should_alert"] and result["label"] in _RISK_ALERT_MESSAGES:
        info = _RISK_ALERT_MESSAGES[result["label"]]
        _fire_alert(info["caregiver_msg"], info["alert_type"])
        socketio.emit("patient_risk_notification", {
            "class_name":  result["class_name"],
            "hindi_msg":   info["patient_hindi"],
            "confidence":  result["confidence"],
            "severity":    result["severity"],
        })

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO events
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("alert_history", db.fetch_alerts(limit=50))
    emit("status_update",  dict(_system_status))
    if _last_risk_prediction:
        emit("risk_update", {
            "label":       _last_risk_prediction.get("label", 0),
            "class_name":  _last_risk_prediction.get("class_name", "Normal"),
            "confidence":  _last_risk_prediction.get("confidence", 0),
            "severity":    _last_risk_prediction.get("severity", "none"),
            "probs":       _last_risk_prediction.get("probabilities", {}),
            "overridden":  _last_risk_prediction.get("overridden_to_false_alarm", False),
            "timestamp":   time.strftime("%H:%M:%S"),
        })


@socketio.on("caregiver_message")
def on_caregiver_message(data):
    """
    Caregiver sends a message.
    Broadcast to ALL clients EXCEPT the sender — so caregiver doesn't
    see a duplicate (they already added the bubble locally in JS).
    """
    socketio.emit("caregiver_message", data, skip_sid=request.sid)


@socketio.on("patient_message")
def on_patient_message(data):
    """
    Patient sends a message back to caregiver.
    Broadcast to all clients except sender.
    """
    socketio.emit("patient_message", data, skip_sid=request.sid)


@socketio.on("sos_trigger")
def on_sos(data):
    patient = data.get("patient", "Patient")
    _fire_alert(f"SOS triggered by {patient}!", "sos")


# ─────────────────────────────────────────────────────────────────────────────
# Periodic status broadcast
# ─────────────────────────────────────────────────────────────────────────────
def _broadcast_status():
    while True:
        time.sleep(3)
        socketio.emit("status_update", dict(_system_status))

threading.Thread(target=_broadcast_status, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser

    print("=" * 60)
    print("  SereneCare + Health Risk Detection")
    print("  Login Page     →  http://127.0.0.1:5000/")
    print("  Caregiver View →  http://127.0.0.1:5000/caregiver")
    print("  Patient View   →  http://127.0.0.1:5000/patient")
    print("  Risk API       →  http://127.0.0.1:5000/risk")
    print("=" * 60)

    def _open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000/")

    threading.Thread(target=_open_browser, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
