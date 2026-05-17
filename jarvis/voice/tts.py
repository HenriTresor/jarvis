"""
Text-to-Speech Module for J.A.R.V.I.S.

Two engines selectable via TTS_ENGINE env var:
  xtts  (default) — Coqui XTTS v2, high quality, supports voice cloning, slow on CPU
  piper            — Piper TTS, near-instant on CPU, preset voices, no cloning

Set TTS_ENGINE=piper in .env for fast responses.
Set PIPER_VOICE=<voice-name> to choose a Piper voice (default: en_GB-alan-medium).
Set SPEAKER_WAV=<path> for XTTS voice cloning.
"""

import io
import os
import wave
import numpy as np
import sounddevice as sd
import threading
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class _XTTSEngine:
    """Coqui XTTS v2 — high quality, supports voice cloning, slow on CPU."""

    SAMPLE_RATE = 24000
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_SPEAKER = "Ana Florence"

    def __init__(self, speaker_wav: Optional[str] = None) -> None:
        import torch
        _orig = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig(*a, **kw)
        torch.load = _patched

        from TTS.api import TTS
        print("[TTS] Loading Coqui XTTS v2...")
        print("[TTS] (First run may take 2-5 minutes to download model)")
        self.tts = TTS(self.MODEL_NAME)
        self.sample_rate = self.SAMPLE_RATE
        self.speaker_wav: Optional[str] = None

        if speaker_wav and os.path.exists(speaker_wav):
            self.speaker_wav = speaker_wav
            print(f"[TTS] Voice cloning enabled: {speaker_wav}")
        else:
            if speaker_wav:
                print(f"[TTS] Warning: SPEAKER_WAV not found: {speaker_wav} — using default voice.")
            print(f"[TTS] Using default voice: {self.DEFAULT_SPEAKER}")

    def synthesize(self, text: str):
        if self.speaker_wav:
            wav = self.tts.tts(text=text, speaker_wav=self.speaker_wav, language="en")
        else:
            wav = self.tts.tts(text=text, speaker=self.DEFAULT_SPEAKER, language="en")
        return np.array(wav, dtype=np.float32), self.sample_rate


class _PiperEngine:
    """Piper TTS — near-instant on CPU, preset voices, no voice cloning."""

    def __init__(self, voice: str = "en_GB-alan-medium") -> None:
        from piper import PiperVoice
        model_path = self._get_model(voice)
        print(f"[TTS] Loading Piper voice: {voice}")
        self.voice = PiperVoice.load(
            model_path,
            config_path=model_path + ".json",
            use_cuda=False,
        )
        self.sample_rate: int = self.voice.config.sample_rate

    def _get_model(self, voice: str) -> str:
        cache_dir = os.path.expanduser("~/.local/share/jarvis/piper")
        os.makedirs(cache_dir, exist_ok=True)

        parts = voice.split("-")
        lang_region = parts[0]           # e.g. en_US
        lang = lang_region.split("_")[0] # e.g. en
        name = "-".join(parts[1:-1])     # e.g. ryan (handles multi-word names)
        quality = parts[-1]              # e.g. high
        folder = f"{lang}/{lang_region}/{name}/{quality}"

        onnx_path = os.path.join(cache_dir, folder, f"{voice}.onnx")
        json_path = onnx_path + ".json"

        if not os.path.exists(onnx_path) or not os.path.exists(json_path):
            print(f"[TTS] Downloading Piper model: {voice} (one-time)...")
            from huggingface_hub import hf_hub_download
            for filename in [f"{voice}.onnx", f"{voice}.onnx.json"]:
                hf_hub_download(
                    repo_id="rhasspy/piper-voices",
                    filename=f"{folder}/{filename}",
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
            print("[TTS] Piper model ready.")

        return onnx_path

    def synthesize(self, text: str):
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            print("[TTS] Piper produced no audio — check model file")
            return np.zeros(1, dtype=np.float32), self.sample_rate
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return audio.astype(np.float32), self.sample_rate


class TextToSpeech:
    """
    TTS facade — delegates to Piper (fast) or XTTS (quality/cloning).

    .env options:
      TTS_ENGINE=piper          use Piper (fast, CPU-friendly)
      TTS_ENGINE=xtts           use XTTS v2 (default, supports voice cloning)
      PIPER_VOICE=en_GB-alan-medium   Piper voice name
      SPEAKER_WAV=/path/to/sample.wav   XTTS voice cloning source
    """

    def __init__(self, speaker_wav: Optional[str] = None) -> None:
        engine = os.getenv("TTS_ENGINE", "xtts").lower()
        speaker_wav = speaker_wav or os.getenv("SPEAKER_WAV") or None

        if engine == "piper":
            voice = os.getenv("PIPER_VOICE", "en_GB-alan-medium")
            self._engine = _PiperEngine(voice=voice)
        else:
            self._engine = _XTTSEngine(speaker_wav=speaker_wav)

        print("[TTS] TTS engine ready.")

    def speak(self, text: str) -> None:
        try:
            if not text or not text.strip():
                return
            audio, sr = self._engine.synthesize(text)
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Error in speak: {e}")

    def generate_audio_bytes(self, text: str) -> bytes:
        try:
            if not text or not text.strip():
                return b""
            audio, sr = self._engine.synthesize(text)
            audio_int16 = (audio * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            print(f"[TTS] Audio bytes generated ({len(buf.getvalue())} bytes).")
            return buf.getvalue()
        except Exception as e:
            print(f"[TTS] Error in generate_audio_bytes: {e}")
            return b""

    def speak_async(self, text: str) -> None:
        try:
            if not text or not text.strip():
                return
            threading.Thread(target=self.speak, args=(text,), daemon=True).start()
        except Exception as e:
            print(f"[TTS] Error in speak_async: {e}")
