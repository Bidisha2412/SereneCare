"""
health_risk_model.py
====================
SereneCare Health Risk Detection Engine

Trains a Random Forest on synthetic joint-movement data and saves:
  - health_risk_model.pkl   (trained RandomForest)
  - health_risk_scaler.pkl  (StandardScaler)

Predicts 4 classes:
  0 = Normal Movement
  1 = Heart Attack Risk
  2 = Panic Attack
  3 = False Alarm

Call from anywhere:
    from health_risk_model import HealthRiskPredictor
    predictor = HealthRiskPredictor()
    result = predictor.predict_from_motion(motion_features)
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Normal', 'Heart Attack Risk', 'Panic Attack', 'False Alarm']
CLASS_LABELS = [0, 1, 2, 3]

FEATURE_COLS = [
    'joint_velocity', 'acceleration', 'tremor_index',
    'posture_angle', 'movement_variance', 'heart_rate_sim',
    'lateral_sway', 'limb_asymmetry',
    'kinetic_energy', 'jerk_approx', 'risk_composite',
    'stability_score', 'hr_variance_proxy'
]

MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'health_risk_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'health_risk_scaler.pkl')

CONFIDENCE_THRESHOLD = 0.65   # below this → reclassify as False Alarm


# ── Dataset Generation ───────────────────────────────────────────────────────
def _generate_dataset(n_samples=3000):
    data = []
    per  = n_samples // 4

    # Normal (0)
    for _ in range(per):
        data.append({
            'joint_velocity':    np.random.normal(0.30, 0.05),
            'acceleration':      np.random.normal(0.20, 0.05),
            'tremor_index':      np.random.beta(1, 9),
            'posture_angle':     np.random.normal(5, 3),
            'movement_variance': np.random.normal(0.10, 0.02),
            'heart_rate_sim':    np.random.normal(72, 8),
            'lateral_sway':      np.random.normal(2, 0.5),
            'limb_asymmetry':    np.random.beta(1, 9),
            'label': 0
        })

    # Heart Attack Risk (1)
    # Near-freeze velocity, sudden jerk, leaning posture, tachycardia
    for _ in range(per):
        data.append({
            'joint_velocity':    np.random.normal(0.05, 0.03),
            'acceleration':      np.random.normal(0.60, 0.15),
            'tremor_index':      np.random.beta(3, 4),
            'posture_angle':     np.random.normal(40, 10),
            'movement_variance': np.random.normal(0.45, 0.10),
            'heart_rate_sim':    np.random.normal(115, 20),
            'lateral_sway':      np.random.normal(12, 3),
            'limb_asymmetry':    np.random.beta(6, 2),
            'label': 1
        })

    # Panic Attack (2)
    # Rapid chaotic motion, very high HR, high tremor, erratic variance
    for _ in range(per):
        data.append({
            'joint_velocity':    np.random.normal(0.80, 0.15),
            'acceleration':      np.random.normal(1.10, 0.20),
            'tremor_index':      np.random.beta(5, 3),
            'posture_angle':     np.random.normal(12, 5),
            'movement_variance': np.random.normal(0.70, 0.12),
            'heart_rate_sim':    np.random.normal(130, 18),
            'lateral_sway':      np.random.normal(7, 2),
            'limb_asymmetry':    np.random.beta(2, 5),
            'label': 2
        })

    # False Alarm (3)
    # High motion (exercise/stumble) but normal-ish HR, symmetric limbs
    for _ in range(per):
        data.append({
            'joint_velocity':    np.random.normal(0.90, 0.20),
            'acceleration':      np.random.normal(0.90, 0.20),
            'tremor_index':      np.random.beta(1, 7),
            'posture_angle':     np.random.normal(25, 10),
            'movement_variance': np.random.normal(0.55, 0.10),
            'heart_rate_sim':    np.random.normal(90, 12),
            'lateral_sway':      np.random.normal(5, 1.5),
            'limb_asymmetry':    np.random.beta(1.5, 6),
            'label': 3
        })

    df = pd.DataFrame(data)
    df['joint_velocity']    = df['joint_velocity'].clip(0.0, 3.0)
    df['acceleration']      = df['acceleration'].clip(0.0, 3.0)
    df['tremor_index']      = df['tremor_index'].clip(0.0, 1.0)
    df['posture_angle']     = df['posture_angle'].clip(0.0, 90.0)
    df['movement_variance'] = df['movement_variance'].clip(0.0, 1.0)
    df['heart_rate_sim']    = df['heart_rate_sim'].clip(40, 200)
    df['lateral_sway']      = df['lateral_sway'].clip(0.0, 30.0)
    df['limb_asymmetry']    = df['limb_asymmetry'].clip(0.0, 1.0)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Feature Engineering ──────────────────────────────────────────────────────
def _engineer(df):
    eps = 1e-6
    df = df.copy()
    df['kinetic_energy']    = 0.5 * df['joint_velocity'] ** 2
    df['jerk_approx']       = df['acceleration'] / (df['joint_velocity'] + eps)
    df['risk_composite']    = df['tremor_index'] * df['heart_rate_sim'] / 100
    df['stability_score']   = 1.0 / (df['posture_angle'] * df['lateral_sway'] + 1)
    df['hr_variance_proxy'] = df['heart_rate_sim'] * df['movement_variance']
    return df


# ── Train + Save ─────────────────────────────────────────────────────────────
def train_and_save():
    print("[HealthRisk] Generating dataset…")
    df = _generate_dataset(3000)
    df = _engineer(df)

    X = df[FEATURE_COLS]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("[HealthRisk] Training Random Forest (200 trees)…")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"[HealthRisk] Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES,
                                 zero_division=0, digits=3))

    with open(MODEL_PATH,  'wb') as f: pickle.dump(model,  f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    print(f"[HealthRisk] Model saved → {MODEL_PATH}")
    print(f"[HealthRisk] Scaler saved → {SCALER_PATH}")
    return model, scaler


# ── Predictor Class (used by app.py) ────────────────────────────────────────
class HealthRiskPredictor:
    """
    Load once at startup; call predict_from_motion() on every frame batch.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print("[HealthRisk] Model not found — training now…")
            self.model, self.scaler = train_and_save()
        else:
            with open(MODEL_PATH,  'rb') as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f: self.scaler = pickle.load(f)
            print("[HealthRisk] Model loaded from disk.")

        self._history      = []   # rolling window of raw features
        self._last_result  = None
        self._cooldown     = {}   # prevent alert spam per class

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_from_motion(self, motion_features: dict) -> dict:
        """
        Takes a dict of raw motion signals derived from OpenCV.
        Returns a prediction result dict.

        motion_features keys (all floats):
          joint_velocity, acceleration, tremor_index, posture_angle,
          movement_variance, heart_rate_sim, lateral_sway, limb_asymmetry
        """
        # Build engineered row
        row = self._engineer_single(motion_features)
        X   = np.array([[row[c] for c in FEATURE_COLS]])
        X_s = self.scaler.transform(X)

        probs     = self.model.predict_proba(X_s)[0]
        raw_pred  = int(np.argmax(probs))
        confidence= float(probs[raw_pred])

        # False-alarm filter — demote low-confidence medical alerts
        if raw_pred in (1, 2) and confidence < CONFIDENCE_THRESHOLD:
            final_pred  = 3   # False Alarm
            overridden  = True
        else:
            final_pred  = raw_pred
            overridden  = False

        result = {
            'label':       int(final_pred),
            'class_name':  CLASS_NAMES[final_pred],
            'confidence':  round(confidence * 100, 1),
            'probabilities': {
                CLASS_NAMES[i]: round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            },
            'raw_prediction': CLASS_NAMES[raw_pred],
            'overridden_to_false_alarm': overridden,
            'should_alert': self._should_alert(final_pred, confidence),
            'severity':    self._severity(final_pred, confidence),
            'features':    motion_features,
        }

        self._last_result = result
        return result

    def get_last_result(self):
        return self._last_result

    # ── Internal ─────────────────────────────────────────────────────────────

    def _engineer_single(self, f: dict) -> dict:
        eps = 1e-6
        r   = dict(f)
        r['kinetic_energy']    = 0.5 * r['joint_velocity'] ** 2
        r['jerk_approx']       = r['acceleration'] / (r['joint_velocity'] + eps)
        r['risk_composite']    = r['tremor_index'] * r['heart_rate_sim'] / 100
        r['stability_score']   = 1.0 / (r['posture_angle'] * r['lateral_sway'] + 1)
        r['hr_variance_proxy'] = r['heart_rate_sim'] * r['movement_variance']
        return r

    def _should_alert(self, label: int, confidence: float) -> bool:
        """Only alert for classes 1 and 2 with enough confidence, with cooldown."""
        if label not in (1, 2):
            return False
        now = __import__('time').time()
        last = self._cooldown.get(label, 0)
        if now - last < 30:          # 30-second cooldown per class
            return False
        self._cooldown[label] = now
        return True

    def _severity(self, label: int, confidence: float) -> str:
        if label == 0:                                return 'none'
        if label == 3:                                return 'low'
        if label == 1 and confidence >= 0.80:         return 'critical'
        if label == 1 and confidence >= 0.65:         return 'high'
        if label == 2 and confidence >= 0.75:         return 'high'
        if label == 2 and confidence >= 0.65:         return 'medium'
        return 'low'


# ── Motion Feature Extractor (used inside monitoring loop) ──────────────────
class MotionFeatureExtractor:
    """
    Converts raw OpenCV frame data into structured features
    that HealthRiskPredictor understands.

    Call update(frame, motion, fall) every frame.
    Call get_features() every ~2 seconds to get a prediction-ready dict.
    """

    def __init__(self, window=30):
        self._window      = window   # frames
        self._velocities  = []
        self._frame_prev  = None
        self._sway_vals   = []
        self._step        = 0
        # simulated vitals (in real deployment replace with sensor data)
        self._hr_base     = 72.0
        self._hr_current  = 72.0

    def update(self, frame, motion: bool, fall: bool, inactive_duration: float):
        import cv2
        self._step += 1

        # ── Joint velocity proxy: mean absolute frame diff ────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._frame_prev is not None:
            diff = cv2.absdiff(gray, self._frame_prev).astype(float)
            vel  = float(diff.mean()) / 255.0 * 3.0   # normalize to ~m/s
        else:
            vel = 0.0
        self._velocities.append(vel)
        if len(self._velocities) > self._window:
            self._velocities.pop(0)
        self._frame_prev = gray

        # ── Lateral sway proxy: optical flow centroid shift ───────────────
        self._sway_vals.append(vel * 10.0)   # simplified proxy
        if len(self._sway_vals) > self._window:
            self._sway_vals.pop(0)

        # ── Simulated HR: rises with motion, rises more with inactivity ──
        if motion:
            self._hr_current = min(200, self._hr_current + np.random.uniform(0.2, 1.2))
        else:
            self._hr_current = max(50, self._hr_current - np.random.uniform(0.1, 0.5))
        if inactive_duration > 45:
            self._hr_current += np.random.uniform(0, 0.8)   # mild anxiety proxy

    def get_features(self, fall: bool, inactive_duration: float) -> dict:
        vels = self._velocities if self._velocities else [0.0]
        vel_arr = np.array(vels)

        vel_mean  = float(vel_arr.mean())
        vel_std   = float(vel_arr.std()) if len(vel_arr) > 1 else 0.0
        accel     = float(np.abs(np.diff(vel_arr)).mean()) if len(vel_arr) > 1 else 0.0
        sway_mean = float(np.mean(self._sway_vals)) if self._sway_vals else 0.0

        # Tremor index — high-freq oscillation proxy (std of velocity diffs)
        diffs       = np.diff(vel_arr) if len(vel_arr) > 2 else np.array([0.0])
        tremor_idx  = float(np.std(diffs))
        tremor_idx  = min(1.0, tremor_idx * 10)

        # Posture angle proxy — fall → high angle, inactivity → rising
        posture = 40.0 if fall else min(60.0, inactive_duration * 0.5 + 5.0)

        # Limb asymmetry proxy — asymmetric motion in left/right halves
        asymmetry = min(1.0, vel_std / (vel_mean + 1e-6))

        return {
            'joint_velocity':    round(max(0.0, vel_mean), 4),
            'acceleration':      round(max(0.0, accel), 4),
            'tremor_index':      round(tremor_idx, 4),
            'posture_angle':     round(posture, 2),
            'movement_variance': round(min(1.0, vel_std), 4),
            'heart_rate_sim':    round(self._hr_current, 1),
            'lateral_sway':      round(min(30.0, sway_mean), 3),
            'limb_asymmetry':    round(asymmetry, 4),
        }


# ── Standalone training ───────────────────────────────────────────────────────
if __name__ == '__main__':
    train_and_save()
    print("\n[HealthRisk] Quick self-test:")
    p = HealthRiskPredictor()
    test_cases = [
        {'joint_velocity':0.30,'acceleration':0.20,'tremor_index':0.05,
         'posture_angle':5,'movement_variance':0.10,'heart_rate_sim':72,
         'lateral_sway':2,'limb_asymmetry':0.05},
        {'joint_velocity':0.04,'acceleration':0.65,'tremor_index':0.55,
         'posture_angle':42,'movement_variance':0.48,'heart_rate_sim':118,
         'lateral_sway':13,'limb_asymmetry':0.75},
        {'joint_velocity':0.82,'acceleration':1.15,'tremor_index':0.68,
         'posture_angle':11,'movement_variance':0.72,'heart_rate_sim':132,
         'lateral_sway':7.5,'limb_asymmetry':0.30},
        {'joint_velocity':0.95,'acceleration':0.90,'tremor_index':0.10,
         'posture_angle':26,'movement_variance':0.56,'heart_rate_sim':92,
         'lateral_sway':5.2,'limb_asymmetry':0.18},
    ]
    expected = ['Normal', 'Heart Attack Risk', 'Panic Attack', 'False Alarm']
    for tc, exp in zip(test_cases, expected):
        r = p.predict_from_motion(tc)
        status = '✓' if r['class_name'] == exp else '✗'
        print(f"  {status} Expected: {exp:20s} | Got: {r['class_name']:20s} | "
              f"Confidence: {r['confidence']}%")
