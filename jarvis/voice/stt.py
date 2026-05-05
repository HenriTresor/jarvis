"""
Speech-to-Text Module for J.A.R.V.I.S.

Transcribes audio to text using faster-whisper (local, free).
Model runs on CPU in int8 mode for speed and low memory.
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional


class SpeechToText:
    """
    Records audio and transcribes it to text using faster-whisper.

    Records until silence is detected (configurable threshold/duration).
    Transcribes using a local Whisper model (no API calls).
    
    Free tool: faster-whisper (CTranslate2 port of OpenAI Whisper)
    Model sizes: tiny (~80MB), base (~150MB), medium (~500MB), large (~1.5GB)
    
    Recommended: 'base' for balanced speed/accuracy on CPU.
    On first load, downloads the model (~150MB for base).
    
    Example:
        stt = SpeechToText(model_size="base")
        audio = stt.record_until_silence()
        text = stt.transcribe(audio)
        # or all at once:
        text = stt.listen()
    """

    SAMPLE_RATE: int = 16000
    SILENCE_THRESHOLD: float = 0.01
    SILENCE_DURATION: float = 1.5  # seconds of silence = end of speech

    def __init__(self, model_size: str = "base") -> None:
        """
        Initialize the Speech-to-Text engine.

        Downloads and loads the faster-whisper model on first run.
        Uses CPU with int8 quantization for speed.

        Args:
            model_size: Whisper model size. Options:
                'tiny', 'base', 'small', 'medium', 'large'
                Larger = more accurate but slower.

        Raises:
            Exception: If model cannot be loaded
        """
        try:
            print(f"[STT] Loading Whisper model: {model_size}")
            print(f"[STT] (First run may take 1-2 minutes to download model)")

            # device="cpu", compute_type="int8" = fastest on CPU
            # compute_type="float32" on GPU or if you need maximum accuracy
            self.model: WhisperModel = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8"
            )
            print(f"[STT] Whisper model '{model_size}' loaded successfully.")
        except Exception as e:
            print(f"[STT] Error in __init__: {e}")
            raise

    def record_until_silence(self, max_seconds: int = 15) -> np.ndarray:
        """
        Records audio from the microphone until silence is detected.

        Listens for speech, then stops recording after detecting silence
        longer than SILENCE_DURATION. Useful for hands-free operation.

        Args:
            max_seconds: Maximum recording duration (safety limit).
                If no silence detected, stops after this time.

        Returns:
            Numpy array of concatenated audio frames (mono, float32, 16kHz)

        Raises:
            Exception: If microphone cannot be accessed
        """
        try:
            print(f"[STT] Recording... (speak now)")
            frames: list = []
            silence_frames: int = 0

            # Calculate how many consecutive silent frames = end of speech
            silence_limit: int = int(
                self.SILENCE_DURATION * self.SAMPLE_RATE / 512
            )

            # Calculate total frames to record (safety limit)
            total_frames: int = int(max_seconds * self.SAMPLE_RATE / 512)

            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=512
            ) as stream:
                for frame_idx in range(total_frames):
                    try:
                        data, _ = stream.read(512)
                        frames.append(data[:, 0])

                        # Measure volume of this frame
                        volume: float = np.abs(data).mean()

                        # Update silence counter
                        if volume < self.SILENCE_THRESHOLD:
                            silence_frames += 1
                            if silence_frames >= silence_limit:
                                print(f"[STT] Silence detected. Stopping recording.")
                                break
                        else:
                            silence_frames = 0
                    except Exception as e:
                        print(f"[STT] Error reading audio frame: {e}")
                        break

            # Concatenate all frames into a single array
            audio: np.ndarray = np.concatenate(frames)
            print(f"[STT] Recording complete. Duration: {len(audio) / self.SAMPLE_RATE:.1f}s")
            return audio
        except Exception as e:
            print(f"[STT] Error in record_until_silence: {e}")
            return np.array([], dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribes a numpy audio array to text.

        Uses the loaded Whisper model to generate transcript.
        Returns the transcribed text, or empty string on error.

        Args:
            audio: Numpy array (float32, mono, 16kHz)

        Returns:
            Transcribed text as string. Empty string if transcription fails.
        """
        try:
            if len(audio) == 0:
                print(f"[STT] Error: Empty audio array")
                return ""

            print(f"[STT] Transcribing... (this may take a moment)")

            # Run transcription
            segments, info = self.model.transcribe(
                audio,
                beam_size=5,
                language="en",
                condition_on_previous_text=False
            )

            # Collect all segment text
            text: str = " ".join([seg.text for seg in segments]).strip()
            print(f"[STT] Transcribed: '{text}'")
            return text
        except Exception as e:
            print(f"[STT] Error in transcribe: {e}")
            return ""

    def listen(self) -> str:
        """
        Full pipeline: record audio then transcribe it.

        Convenience method combining record_until_silence() and transcribe().
        Use this for simple cases; use the individual methods for more control.

        Returns:
            Transcribed text as string. Empty string if recording/transcription fails.
        """
        try:
            audio: np.ndarray = self.record_until_silence()
            text: str = self.transcribe(audio)
            return text
        except Exception as e:
            print(f"[STT] Error in listen: {e}")
            return ""
