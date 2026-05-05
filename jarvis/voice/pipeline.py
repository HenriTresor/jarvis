"""
Voice Pipeline Orchestrator for J.A.R.V.I.S.

Ties together wake word detection → STT → LLM brain → TTS.
Implements the full voice interaction loop: listen for wake word,
record user command, process through brain, and speak response.
"""

import time
import threading
from typing import Callable, Optional
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
        speaker_wav: Optional[str] = None
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

            # Greet the user
            self.tts.speak("Yes?")

            # Record the user's command
            print(f"[Pipeline] Recording command...")
            user_text: str = self.stt.listen()

            # Check if transcription was successful
            if not user_text or not user_text.strip():
                print(f"[Pipeline] No speech detected.")
                self.tts.speak("Sorry, I didn't catch that.")
                self._processing = False
                return

            # Send command to the brain (LLM + agent)
            print(f"[Pipeline] Sending to brain: '{user_text}'")
            response: str = self.brain(user_text)

            # Check if brain returned a valid response
            if not response or not response.strip():
                print(f"[Pipeline] Brain returned empty response.")
                response = "I'm not sure how to respond to that."

            # Speak the response
            self.tts.speak(response)

            print(f"[Pipeline] Interaction complete.")
            self._processing = False
        except Exception as e:
            print(f"[Pipeline] Error in _on_wake: {e}")
            self._processing = False

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
