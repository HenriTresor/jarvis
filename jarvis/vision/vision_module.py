"""
Vision Module for J.A.R.V.I.S.

Computer vision capabilities using LLaVA (multimodal LLM via Ollama).
Enables image analysis, text reading, and motion detection.

Free tool: LLaVA 7B via Ollama — runs locally, ~4.5GB download.
"""

import cv2
import base64
import io
import ollama
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Optional


class VisionModule:
    """
    Computer vision for Jarvis using LLaVA (local multimodal LLM).

    Capabilities:
    - Describe what the camera sees (object detection, scene analysis)
    - Read text in images (OCR-like functionality)
    - Identify objects, people, and activities in frame
    - Analyze documents, whiteboards, screens
    - Simple motion detection

    Free tool: LLaVA 7B — runs locally via Ollama (~4.5GB model).
    No API calls, no keys needed.

    Example:
        vision = VisionModule(camera_index=0)

        # Describe what's happening on camera
        description = vision.describe_environment()
        print(description)  # "There is a desk with a computer and..."

        # Read text from an image file
        text = vision.read_text_in_frame()
        print(text)  # "Technical Documentation v1.0"

        # Analyze a specific image with custom question
        analysis = vision.analyze_image_file(
            "documents/whiteboard.jpg",
            prompt="What are the key action items?"
        )
    """

    MODEL: str = "llava:7b"
    SAMPLE_RATE: int = 24000

    def __init__(self, camera_index: int = 0) -> None:
        """
        Initialize the vision module.

        Verifies that LLaVA model is available via Ollama.

        Args:
            camera_index: Webcam device index (0 = default camera)

        Raises:
            Exception: If LLaVA model cannot be accessed via Ollama
        """
        try:
            self.camera_index: int = camera_index
            print(f"[Vision] Initializing vision module (model: {self.MODEL})")

            # Verify LLaVA is available
            models_resp = ollama.list()
            if isinstance(models_resp, dict):
                model_list = models_resp.get('models', [])
            else:
                model_list = getattr(models_resp, 'models', []) or []
            model_names: list = []
            for m in model_list:
                if isinstance(m, dict):
                    name = m.get('name', '') or m.get('model', '')
                else:
                    name = getattr(m, 'model', '') or getattr(m, 'name', '')
                model_names.append(name.split(':')[0])

            if "llava" not in model_names:
                print(f"[Vision] Warning: LLaVA not found in Ollama models")
                print(f"[Vision] Pull it with: ollama pull {self.MODEL}")

            print(f"[Vision] Vision module ready.")
        except Exception as e:
            print(f"[Vision] Error in __init__: {e}")
            raise

    def _capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the webcam.

        Returns:
            Numpy array (BGR format from OpenCV)

        Raises:
            Exception: If camera cannot be accessed
        """
        try:
            print(f"[Vision] Capturing frame from camera {self.camera_index}")
            cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera_index}"
                )

            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Could not read frame from camera")

            return frame
        except Exception as e:
            print(f"[Vision] Error in _capture_frame: {e}")
            raise

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert a numpy BGR frame to base64-encoded JPEG.

        Args:
            frame: Numpy array (BGR format from OpenCV)

        Returns:
            Base64-encoded JPEG string

        Raises:
            Exception: On image encoding error (caught internally)
        """
        try:
            # Convert BGR to RGB for PIL
            rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image: Image.Image = Image.fromarray(rgb_frame)

            # Encode to JPEG in memory
            buffer: io.BytesIO = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)

            # Return base64 string
            b64_str: str = base64.b64encode(buffer.getvalue()).decode()
            return b64_str
        except Exception as e:
            print(f"[Vision] Error in _frame_to_base64: {e}")
            raise

    def _ask_llava(self, image_b64: str, prompt: str) -> str:
        """
        Send an image and prompt to LLaVA for analysis.

        Args:
            image_b64: Base64-encoded image data
            prompt: Question or instruction for LLaVA

        Returns:
            LLaVA's analysis/response as string

        Raises:
            Exception: On LLaVA API error (caught internally)
        """
        try:
            if not prompt or not prompt.strip():
                return "Error: No prompt provided"

            print(f"[Vision] Sending image to LLaVA...")

            response: dict = ollama.chat(
                model=self.MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }]
            )

            result: str = response["message"]["content"]
            print(f"[Vision] LLaVA response: {result[:100]}...")
            return result
        except Exception as e:
            print(f"[Vision] Error in _ask_llava: {e}")
            return f"Vision analysis error: {e}"

    def describe_environment(self) -> str:
        """
        Capture and describe what the camera sees.

        Uses LLaVA to analyze the current environment: people, objects,
        activities, layout, mood, etc.

        Returns:
            Text description of the environment

        Raises:
            Exception: On camera or vision error (caught internally)
        """
        try:
            print(f"[Vision] Describing environment...")
            frame: np.ndarray = self._capture_frame()
            image_b64: str = self._frame_to_base64(frame)

            prompt: str = (
                "Describe what you see in this image concisely. "
                "Focus on people, objects, the environment, and what's happening. "
                "Be specific and practical."
            )

            description: str = self._ask_llava(image_b64, prompt)
            return description
        except Exception as e:
            print(f"[Vision] Error in describe_environment: {e}")
            return f"Error describing environment: {e}"

    def read_text_in_frame(self) -> str:
        """
        Extract and read all visible text from the current camera frame.

        Uses LLaVA for OCR-like text reading from the image.

        Returns:
            Transcribed text, or message if no text found

        Raises:
            Exception: On camera or vision error (caught internally)
        """
        try:
            print(f"[Vision] Reading text in frame...")
            frame: np.ndarray = self._capture_frame()
            image_b64: str = self._frame_to_base64(frame)

            prompt: str = (
                "Read and transcribe all text visible in this image. "
                "Include the exact text you see, preserving formatting if possible. "
                "If no text is present, say 'No text found'."
            )

            text: str = self._ask_llava(image_b64, prompt)
            return text
        except Exception as e:
            print(f"[Vision] Error in read_text_in_frame: {e}")
            return f"Error reading text: {e}"

    def analyze_image_file(
        self,
        path: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Analyze an image file with an optional custom prompt.

        Args:
            path: Path to image file (e.g., "documents/screen.png")
            prompt: Custom prompt (e.g., "What information is shown?")
                   If None, uses generic description prompt

        Returns:
            LLaVA's analysis of the image

        Raises:
            Exception: On file read or vision error (caught internally)
        """
        try:
            if not path or not path.strip():
                return "Error: No file path provided"

            print(f"[Vision] Analyzing image file: {path}")

            # Read image file
            with open(path, "rb") as f:
                image_data: bytes = f.read()

            # Convert to base64
            image_b64: str = base64.b64encode(image_data).decode()

            # Use provided prompt or default
            if not prompt or not prompt.strip():
                prompt = "Analyze this image in detail."

            analysis: str = self._ask_llava(image_b64, prompt)
            return analysis
        except FileNotFoundError:
            error_msg: str = f"File not found: {path}"
            print(f"[Vision] {error_msg}")
            return error_msg
        except Exception as e:
            print(f"[Vision] Error in analyze_image_file: {e}")
            return f"Error analyzing image: {e}"

    def detect_motion(self, threshold: int = 500) -> bool:
        """
        Detect if motion is present on the camera.

        Simple frame differencing algorithm:
        1. Capture two consecutive frames
        2. Compute absolute difference
        3. Count non-zero pixels above threshold
        4. Return True if motion detected

        Args:
            threshold: Pixel count threshold for motion detection (default: 500)

        Returns:
            True if motion detected, False otherwise

        Raises:
            Exception: On camera error (caught internally, returns False)
        """
        try:
            print(f"[Vision] Detecting motion (threshold: {threshold} pixels)...")

            cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera_index}"
                )

            # Capture first frame
            ret1: bool
            frame1: np.ndarray
            ret1, frame1 = cap.read()

            if not ret1:
                cap.release()
                raise RuntimeError("Could not read first frame")

            # Capture second frame
            ret2: bool
            frame2: np.ndarray
            ret2, frame2 = cap.read()

            cap.release()

            if not ret2:
                raise RuntimeError("Could not read second frame")

            # Compute difference
            diff: np.ndarray = cv2.absdiff(frame1, frame2)

            # Convert to grayscale
            gray: np.ndarray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Threshold
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

            # Count non-zero pixels (motion area)
            motion_area: int = cv2.countNonZero(thresh)

            motion_detected: bool = motion_area > threshold
            print(f"[Vision] Motion area: {motion_area} pixels")
            print(f"[Vision] Motion detected: {motion_detected}")

            return motion_detected
        except Exception as e:
            print(f"[Vision] Error in detect_motion: {e}")
            return False

    def take_screenshot(self, output_path: str = None) -> str:
        """
        Capture a screenshot from the camera and save it.

        Args:
            output_path: Path to save the screenshot
                        If None, saves to ~/jarvis_files/screenshots/ with timestamp

        Returns:
            Path to saved screenshot file, or error message

        Raises:
            Exception: On file write error (caught internally)
        """
        try:
            print(f"[Vision] Taking screenshot...")

            frame: np.ndarray = self._capture_frame()

            # Determine output path
            if not output_path:
                import os
                screenshots_dir: str = os.path.expanduser(
                    "~/jarvis_files/screenshots"
                )
                os.makedirs(screenshots_dir, exist_ok=True)

                timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(screenshots_dir, f"{timestamp}.jpg")

            # Save the frame
            success: bool = cv2.imwrite(output_path, frame)

            if success:
                print(f"[Vision] Screenshot saved: {output_path}")
                return output_path
            else:
                error_msg: str = f"Failed to save screenshot to {output_path}"
                print(f"[Vision] {error_msg}")
                return error_msg
        except Exception as e:
            print(f"[Vision] Error in take_screenshot: {e}")
            return f"Screenshot error: {e}"
