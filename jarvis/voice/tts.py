"""
Text-to-Speech Module for J.A.R.V.I.S.

Synthesizes speech using Coqui XTTS v2 (local, free).
Supports voice cloning with a 6-second voice sample.
Plays audio through the system speaker.
"""

import numpy as np
import sounddevice as sd
from TTS.api import TTS
import threading
from typing import Optional


class TextToSpeech:
    """
    Converts text to speech locally using Coqui XTTS v2.

    Supports two modes:
    1. Default voice: Uses built-in "Ana Florence" voice
    2. Custom voice cloning: Pass a 6-second voice sample as speaker_wav

    Free tool: Coqui TTS — no API key needed, runs on CPU.
    Model (~1.8GB) downloads on first run. Subsequent runs are instant.
    
    Example (default voice):
        tts = TextToSpeech()
        tts.speak("Hello, sir.")
        tts.speak_async("Processing in the background.")

    Example (voice cloning):
        tts = TextToSpeech(speaker_wav="./my_voice_sample.wav")
        tts.speak("This sounds like me!")
    """

    SAMPLE_RATE: int = 24000
    MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_SPEAKER: str = "Ana Florence"  # built-in voice

    def __init__(self, speaker_wav: Optional[str] = None) -> None:
        """
        Initialize the Text-to-Speech engine.

        Downloads and loads the Coqui XTTS v2 model on first run (~1.8GB).
        Subsequent initializations use cached model (instant).

        Args:
            speaker_wav: Optional path to a 6-second .wav voice sample
                        for voice cloning. If None, uses default voice.

        Raises:
            Exception: If TTS model cannot be loaded or speaker_wav is invalid
        """
        try:
            print(f"[TTS] Loading Coqui XTTS v2...")
            print(f"[TTS] (First run may take 2-5 minutes to download model)")

            self.tts: TTS = TTS(self.MODEL_NAME)
            self.speaker_wav: Optional[str] = speaker_wav
            self.sample_rate: int = self.SAMPLE_RATE

            # Validate speaker_wav if provided
            if self.speaker_wav:
                try:
                    import os
                    if not os.path.exists(self.speaker_wav):
                        raise FileNotFoundError(
                            f"speaker_wav file not found: {self.speaker_wav}"
                        )
                    print(
                        f"[TTS] Voice cloning enabled: {self.speaker_wav}"
                    )
                except Exception as e:
                    print(f"[TTS] Warning: {e}")
                    print(f"[TTS] Falling back to default voice.")
                    self.speaker_wav = None

            print(f"[TTS] TTS engine ready.")
        except Exception as e:
            print(f"[TTS] Error in __init__: {e}")
            raise

    def speak(self, text: str) -> None:
        """
        Synthesizes text and plays it immediately (blocking).

        Generates speech from text using the loaded TTS model and plays
        through the system speaker. This call blocks until playback completes.

        Args:
            text: Text to synthesize and play

        Returns:
            None

        Raises:
            Exception: On TTS synthesis or audio playback error (caught internally)
        """
        try:
            if not text or not text.strip():
                print(f"[TTS] Error: Empty text")
                return

            display_text: str = text[:60] + "..." if len(text) > 60 else text
            print(f"[TTS] Speaking: '{display_text}'")

            # Generate speech
            if self.speaker_wav:
                # Use custom voice cloning
                wav = self.tts.tts(
                    text=text,
                    speaker_wav=self.speaker_wav,
                    language="en"
                )
            else:
                # Use default voice
                wav = self.tts.tts(
                    text=text,
                    speaker=self.DEFAULT_SPEAKER,
                    language="en"
                )

            # Convert to numpy float32 array
            audio: np.ndarray = np.array(wav, dtype=np.float32)

            # Play audio through the system speaker
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()  # Block until playback completes

            print(f"[TTS] Playback complete.")
        except Exception as e:
            print(f"[TTS] Error in speak: {e}")

    def speak_async(self, text: str) -> None:
        """
        Synthesizes text and plays it asynchronously (non-blocking).

        Spawns a daemon thread to synthesize and play the text.
        Returns immediately without waiting for playback to complete.
        Useful for background announcements or processing while speaking.

        Args:
            text: Text to synthesize and play

        Returns:
            None

        Raises:
            Exception: On thread creation error (caught internally)
        """
        try:
            if not text or not text.strip():
                print(f"[TTS] Error: Empty text")
                return

            # Create and start daemon thread
            thread: threading.Thread = threading.Thread(
                target=self.speak,
                args=(text,),
                daemon=True
            )
            thread.start()
            print(f"[TTS] Async speech started (non-blocking).")
        except Exception as e:
            print(f"[TTS] Error in speak_async: {e}")
