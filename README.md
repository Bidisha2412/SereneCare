# CareWatch — Elderly Monitoring System

AI-powered elderly care monitoring system with real-time video analysis, health risk prediction, fall detection, and voice interaction.

## Features

- **Real-time Video Monitoring** — Motion detection and fall detection via webcam
- **Health Risk Prediction** — ML-based detection of heart attack risk, panic attacks
- **Voice AI** — Speech-to-text interaction with elderly patients via Whisper
- **Caregiver Dashboard** — Real-time alerts, video feed, and patient status
- **Patient Dashboard** — Voice interaction, SOS button, caregiver messages
- **Email Notifications** — Automated alerts sent to caregivers

## Project Structure

```
serenecare_v4/
├── app.py                  # Flask entry point
├── config.py               # Pipeline & model configuration
├── requirements.txt        # Python dependencies
│
├── src/                    # Application source code
│   ├── detectors/          # CV-based detection modules
│   │   ├── motion_detection.py
│   │   ├── fall_detection.py
│   │   └── inactivity_monitor.py
│   ├── services/           # Voice AI & notifications
│   │   ├── voice_ai.py
│   │   └── notifier.py
│   └── ml/                 # Machine learning pipeline
│       ├── inference.py        # Real-time inference engine
│       ├── video_pipeline.py   # Video → joint coordinates
│       ├── features.py         # Feature engineering
│       ├── ingestion.py        # Data loading & streaming
│       ├── scaler.py           # Incremental feature scaling
│       ├── trainer.py          # Model training (LightGBM/XGBoost/SGD/LSTM)
│       └── health_risk_model.py # Legacy standalone health risk model
│
├── database/               # SQLite database layer
│   └── db.py
│
├── templates/              # HTML templates (Flask/Jinja2)
│   ├── login.html
│   ├── caregiver_dashboard.html
│   └── patient_dashboard.html
│
├── models/                 # Trained model artifacts (.pkl)
├── output/                 # Training output (models, scalers, cache)
└── training/               # Standalone training scripts
    └── train.py
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

The app opens at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Training

To train the ML model on your own data:

```bash
python -m training.train                # LightGBM (default)
python -m training.train --model xgb    # XGBoost
python -m training.train --model sgd    # SGD (incremental)
python -m training.train --model lstm   # LSTM (GPU recommended)
```

## Tech Stack

- **Backend**: Flask + Flask-SocketIO
- **ML**: LightGBM, XGBoost, scikit-learn, PyTorch (LSTM)
- **CV**: OpenCV, MediaPipe
- **Voice**: OpenAI Whisper, gTTS
- **Database**: SQLite
