"""
voice_ai.py
Records audio → transcribes with Whisper → generates response → speaks via gTTS.
"""

import io
import os
import tempfile
import threading
import time

# Audio recording
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Speech-to-text
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Text-to-speech
try:
    from gtts import gTTS
    import pygame
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ------------------------------------------------------------------ #
# Simple intent → response map                                         #
# ------------------------------------------------------------------ #
INTENTS = {
    # keywords  → response
    ("help", "emergency", "sos", "pain", "hurt"):
        "I have detected you need help. Alerting your caregiver immediately.",

    ("water", "drink", "thirsty"):
        "I heard you need water. Please stay where you are. Help is on the way.",

    ("medicine", "pill", "medication"):
        "Reminding you to take your medicine. Shall I alert the caregiver?",

    ("fall", "fell", "fallen"):
        "I detected a possible fall. Alerting caregiver now. Are you okay?",

    ("bathroom", "toilet"):
        "Understood. Please be careful walking to the bathroom.",

    ("cold", "hot", "temperature"):
        "I will let the caregiver know about your temperature comfort.",

    ("hello", "hi", "hey"):
        "Hello! I am your monitoring assistant. How can I help you today?",
}

DEFAULT_RESPONSE = (
    "I heard you. I am here to help. "
    "If you need urgent assistance please say help or emergency."
)


class VoiceAI:
    def __init__(self, model_size="tiny", sample_rate=16000, duration=5):
        """
        model_size: Whisper model ('tiny', 'base', 'small').
        sample_rate: recording sample rate in Hz.
        duration: seconds to record per interaction.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self._model = None
        self._model_size = model_size
        self._lock = threading.Lock()

        if TTS_AVAILABLE:
            pygame.mixer.init()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def interact(self):
        """
        Full pipeline: record → transcribe → respond → speak.
        Returns (transcription_text, response_text).
        """
        self.speak("I am listening. Please speak now.")
        audio = self._record()
        if audio is None:
            return "[no audio]", "Sorry, I could not access the microphone."

        text = self._transcribe(audio)
        response = self._generate_response(text)
        self.speak(response)
        return text, response

    def speak(self, text):
        """Convert text to speech and play it."""
        if not TTS_AVAILABLE:
            print(f"[TTS] {text}")
            return
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)
                tmp_path = f.name
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"[TTS error] {e}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        with self._lock:
            if self._model is None and WHISPER_AVAILABLE:
                print("[Whisper] Loading model, please wait…")
                self._model = whisper.load_model(self._model_size)
                print("[Whisper] Model ready.")

    def _record(self):
        """Record `duration` seconds of audio. Returns numpy array or None."""
        if not AUDIO_AVAILABLE:
            return None
        try:
            audio = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return audio
        except Exception as e:
            print(f"[Audio record error] {e}")
            return None

    def _transcribe(self, audio):
        """Convert numpy audio array to text via Whisper."""
        self._load_model()
        if self._model is None or audio is None:
            return ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.sample_rate)
                tmp_path = f.name
            result = self._model.transcribe(tmp_path, language="en")
            os.unlink(tmp_path)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[Whisper error] {e}")
            return ""

    def _generate_response(self, text):
        """Match text to an intent and return a response."""
        if not text:
            return DEFAULT_RESPONSE
        lower = text.lower()
        for keywords, response in INTENTS.items():
            if any(kw in lower for kw in keywords):
                return response
        return DEFAULT_RESPONSE
