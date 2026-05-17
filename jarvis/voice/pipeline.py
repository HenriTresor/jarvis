"""
Voice Pipeline Orchestrator for J.A.R.V.I.S.

Ties together wake word detection → STT → LLM brain → TTS.
Implements the full voice interaction loop: listen for wake word,
record user command, process through brain, and speak response.
"""

import subprocess
import time
import threading
from typing import Callable, Optional, Generator
from .wake_word import WakeWordDetector
from .stt import SpeechToText
from .tts import TextToSpeech


class VoicePipeline:
    """
    Complete voice interaction pipeline for J.A.R.V.I.S.

    Orchestrates the full flow:
    1. Continuously listen for wake word (e.g., "Hey Jarvis")
    2. On wake word: record user's command
    3. Transcribe command to text
    4. Send to brain (LLM + agent)
    5. Speak the response
    6. Return to waiting for next wake word

    Usage:
        def my_brain(text: str) -> str:
            # Your LLM / agent here
            return "Response to: " + text

        pipeline = VoicePipeline(brain_callback=my_brain)
        pipeline.start()  # Runs indefinitely (until KeyboardInterrupt or .stop())
    """

    def __init__(
        self,
        brain_callback: Callable[[str], str],
        speaker_wav: Optional[str] = None,
        brain_stream_callback: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the voice pipeline.

        Args:
            brain_callback: Function that takes user text and returns response.
                           Signature: (text: str) -> str
            speaker_wav: Optional path to voice sample for TTS voice cloning

        Raises:
            Exception: If any component (wake word, STT, TTS) fails to initialize
        """
        try:
            self.brain: Callable[[str], str] = brain_callback
            self.brain_stream = brain_stream_callback
            self.stt: SpeechToText = SpeechToText(model_size="base")
            self.tts: TextToSpeech = TextToSpeech(speaker_wav=speaker_wav)
            self.detector: WakeWordDetector = WakeWordDetector(
                on_detected=self._on_wake
            )
            self._active: bool = False
            self._processing: bool = False  # Flag to prevent concurrent processing

            print(f"[Pipeline] Voice pipeline initialized.")
        except Exception as e:
            print(f"[Pipeline] Error in __init__: {e}")
            raise

    def _on_wake(self) -> None:
        """
        Callback invoked when the wake word is detected.

        Triggered by the WakeWordDetector when it recognizes the wake word.
        This method:
        1. Checks if already processing (prevents overlap)
        2. Greets the user
        3. Records the user's command
        4. Sends command to the brain
        5. Speaks the response
        6. Returns to listening
        """
        try:
            # Guard against concurrent processing
            if self._processing:
                return
            self._processing = True

            print(f"[Pipeline] Wake word detected. Processing...")

            # Pause any media playing so it doesn't bleed into STT recording
            was_playing = self._media_pause()

            # Greet the user
            self.tts.speak("Yes?")

            # Record the user's command
            print(f"[Pipeline] Recording command...")
            user_text: str = self.stt.listen()

            # Check if transcription was successful
            if not user_text or not user_text.strip():
                print(f"[Pipeline] No speech detected.")
                self.tts.speak("Sorry, I didn't catch that.")
                if was_playing:
                    self._media_resume()
                self._processing = False
                return

            # Send command to the brain (LLM + agent)
            print(f"[Pipeline] Sending to brain: '{user_text}'")
            if self.brain_stream:
                self._run_streaming_brain(user_text)
            else:
                response: str = self.brain(user_text)
                if not response or not response.strip():
                    response = "I'm not sure how to respond to that."
                self.tts.speak(response)
            if was_playing:
                self._media_resume()

            print(f"[Pipeline] Interaction complete.")
            self._processing = False
        except Exception as e:
            print(f"[Pipeline] Error in _on_wake: {e}")
            self._processing = False

    def _run_streaming_brain(self, user_text: str) -> None:
        """
        Use think_stream to announce what Jarvis is doing immediately,
        then speak the final response once it arrives.
        """
        try:
            stream = self.brain_stream(user_text)
            first_chunk = next(stream, None)
            if first_chunk is None:
                return

            stripped_first = first_chunk.strip()
            # Pre-execution messages are short complete sentences (e.g. "Searching for X.")
            is_pre_exec = stripped_first.endswith(".") and len(stripped_first.split()) <= 8

            if is_pre_exec:
                print(f"[Pipeline] Pre-exec: '{stripped_first}'")
                self.tts.speak(stripped_first)
                remaining = "".join(stream).strip()
                if remaining:
                    self.tts.speak(remaining)
            else:
                # Direct conversational response — collect and speak everything
                full = first_chunk + "".join(stream)
                if full.strip():
                    self.tts.speak(full.strip())
        except Exception as e:
            print(f"[Pipeline] Error in _run_streaming_brain: {e}")

    def _media_pause(self) -> bool:
        """Pause all MPRIS media players. Returns True if something was playing."""
        try:
            status = subprocess.run(
                ["playerctl", "status"],
                capture_output=True, text=True, timeout=2
            )
            playing = status.stdout.strip() == "Playing"
            if playing:
                subprocess.run(["playerctl", "pause"], capture_output=True, timeout=2)
                time.sleep(0.3)  # let audio fully stop before mic opens
                print("[Pipeline] Media paused for recording.")
            return playing
        except FileNotFoundError:
            # playerctl not installed — try dbus-send directly
            try:
                subprocess.run(
                    ["dbus-send", "--dest=org.mpris.MediaPlayer2.spotify",
                     "/org/mpris/MediaPlayer2",
                     "org.mpris.MediaPlayer2.Player.Pause"],
                    capture_output=True, timeout=2
                )
                time.sleep(0.3)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _media_resume(self) -> None:
        """Resume MPRIS media playback."""
        try:
            subprocess.run(["playerctl", "play"], capture_output=True, timeout=2)
            print("[Pipeline] Media resumed.")
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["dbus-send", "--dest=org.mpris.MediaPlayer2.spotify",
                     "/org/mpris/MediaPlayer2",
                     "org.mpris.MediaPlayer2.Player.Play"],
                    capture_output=True, timeout=2
                )
            except Exception:
                pass
        except Exception:
            pass

    def start(self) -> None:
        """
        Start the voice pipeline.

        Begins listening for the wake word and enters the main loop.
        Runs indefinitely until interrupted (KeyboardInterrupt or .stop()).
        This is a blocking call—the main thread will remain in this function.

        Usage:
            pipeline.start()  # Runs forever
        """
        try:
            if self._active:
                print(f"[Pipeline] Pipeline already running.")
                return

            self._active = True
            self.detector.start()
            print(f"[Pipeline] Voice pipeline active. Say 'Hey Jarvis' to begin.")
            print(f"[Pipeline] Press Ctrl+C to stop.")

            # Keep the main thread alive
            try:
                while self._active:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"\n[Pipeline] Keyboard interrupt detected.")
                self.stop()
        except Exception as e:
            print(f"[Pipeline] Error in start: {e}")
            self._active = False

    def stop(self) -> None:
        """
        Stop the voice pipeline gracefully.

        Shuts down the wake word detector and exits the main loop.
        Safe to call multiple times.
        """
        try:
            if not self._active:
                print(f"[Pipeline] Pipeline not running.")
                return

            self._active = False
            self.detector.stop()
            print(f"[Pipeline] Voice pipeline stopped.")
        except Exception as e:
            print(f"[Pipeline] Error in stop: {e}")
