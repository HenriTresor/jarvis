"""
Wake Word Detection Module for J.A.R.V.I.S.

Continuously listens for the wake word using OpenWakeWord.
When detected, fires a callback to trigger audio recording.
Runs on CPU only—no GPU required.
"""

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
import threading
import queue
from typing import Callable, Optional


class WakeWordDetector:
    """
    Listens continuously for the wake word using OpenWakeWord.
    
    When the wake word is detected above the threshold, invokes a callback.
    Uses sounddevice for audio I/O and a background thread for detection.
    
    Free tool: OpenWakeWord — runs fully on CPU, no API key needed.
    
    Example:
        def on_wake():
            print("Wake word detected!")
        
        detector = WakeWordDetector(on_detected=on_wake)
        detector.start()
        # ... detector runs in background ...
        detector.stop()
    """

    SAMPLE_RATE: int = 16000
    CHUNK_DURATION: float = 0.08  # 80ms chunks
    CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        on_detected: Optional[Callable[[], None]] = None,
        threshold: float = 0.5
    ) -> None:
        """
        Initialize the wake word detector.

        Args:
            wake_word: The wake word to listen for (default: "hey_jarvis")
            on_detected: Callback function (no args) invoked when wake word is detected
            threshold: Detection confidence threshold, 0-1 (default: 0.5)

        Raises:
            Exception: If OpenWakeWord model cannot be loaded
        """
        try:
            self.wake_word = wake_word
            self.on_detected = on_detected
            self.threshold = threshold
            self._active = False
            self._queue: queue.Queue = queue.Queue()

            # Load the wake word model from OpenWakeWord
            print(f"[WakeWord] Loading model for '{wake_word}'...")
            self.model: Model = Model(wakeword_models=[wake_word])

            self.stream: Optional[sd.InputStream] = None
            self.detection_thread: Optional[threading.Thread] = None

            print(f"[WakeWord] Model loaded successfully.")
        except Exception as e:
            print(f"[WakeWord] Error in __init__: {e}")
            raise

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """
        Called by sounddevice on each audio chunk.

        Converts float32 audio to int16 and queues for detection.
        This runs in the audio thread—keep it fast.

        Args:
            indata: Audio data as numpy array (float32, mono)
            frames: Number of frames in this chunk
            time_info: Timestamp information from sounddevice
            status: Stream status flags
        """
        try:
            # Extract mono channel and convert to int16 for the model
            audio_chunk = indata[:, 0].copy()  # mono
            chunk_int16 = (audio_chunk * 32767).astype(np.int16)
            self._queue.put(chunk_int16)
        except Exception as e:
            print(f"[WakeWord] Error in _audio_callback: {e}")

    def _detection_loop(self) -> None:
        """
        Detection loop that runs in a daemon thread.

        Continuously reads audio chunks from the queue, runs the model,
        and invokes the callback if the wake word is detected above threshold.
        """
        try:
            print(f"[WakeWord] Detection loop started.")
            while self._active:
                try:
                    # Get chunk with timeout to allow checking _active flag
                    chunk = self._queue.get(timeout=0.1)

                    # Run prediction on the audio chunk
                    predictions: dict = self.model.predict(chunk)

                    # Extract score for the wake word
                    score: float = predictions.get(self.wake_word, 0.0)

                    # If score exceeds threshold, trigger callback
                    if score > self.threshold:
                        print(f"[WakeWord] Detected! Score: {score:.2f}")
                        if self.on_detected:
                            self.on_detected()
                except queue.Empty:
                    # No chunk available yet, continue waiting
                    continue
        except Exception as e:
            print(f"[WakeWord] Error in _detection_loop: {e}")
        finally:
            print(f"[WakeWord] Detection loop stopped.")

    def start(self) -> None:
        """
        Start the wake word detector.

        Begins the audio stream and spawns the detection thread.
        Listens continuously until stop() is called.

        Safe to call multiple times—only starts once.
        """
        try:
            if self._active:
                print(f"[WakeWord] Detector already running.")
                return

            self._active = True

            # Start audio stream with the callback
            self.stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"[WakeWord] Audio stream started.")

            # Start detection thread (daemon so it won't block shutdown)
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True
            )
            self.detection_thread.start()
            print(f"[WakeWord] Listening for '{self.wake_word}'...")
        except Exception as e:
            print(f"[WakeWord] Error in start: {e}")
            self._active = False

    def stop(self) -> None:
        """
        Stop the wake word detector.

        Cleanly shuts down the audio stream and detection thread.
        Safe to call multiple times—only stops if running.
        """
        try:
            if not self._active:
                print(f"[WakeWord] Detector not running.")
                return

            self._active = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                print(f"[WakeWord] Audio stream stopped.")

            if self.detection_thread:
                # Wait for thread to finish (max 2 seconds)
                self.detection_thread.join(timeout=2.0)
                print(f"[WakeWord] Detection thread stopped.")
        except Exception as e:
            print(f"[WakeWord] Error in stop: {e}")
