"""
Vision Module for J.A.R.V.I.S.

Computer vision capabilities using Groq's hosted multimodal model (Llama 4 Scout).
Enables image analysis, text reading, and motion detection — no local model needed.
"""

import cv2
import base64
import io
import os
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class VisionModule:
    """
    Computer vision for Jarvis using Groq's multimodal Llama 4 Scout.

    Capabilities:
    - Describe what the camera sees (object detection, scene analysis)
    - Read text in images (OCR-like functionality)
    - Identify objects, people, and activities in frame
    - Analyze documents, whiteboards, screens
    - Simple motion detection (frame differencing, no LLM needed)

    Requires GROQ_API_KEY in environment. Override model with GROQ_VISION_MODEL.
    """

    MODEL: str = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    def __init__(self, camera_index: int = 0) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("[Vision] GROQ_API_KEY not set. Add it to your .env file.")
        self.client: Groq = Groq(api_key=api_key)
        self.camera_index: int = camera_index
        print(f"[Vision] Vision module ready (model: {self.MODEL})")

    def _capture_frame(self) -> np.ndarray:
        print(f"[Vision] Capturing frame from camera {self.camera_index}")
        cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame from camera")
        return frame

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image: Image.Image = Image.fromarray(rgb_frame)
        buffer: io.BytesIO = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    def _ask_vision(self, image_b64: str, prompt: str) -> str:
        print(f"[Vision] Sending image to Groq ({self.MODEL})...")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
        )
        result: str = response.choices[0].message.content or ""
        print(f"[Vision] Response: {result[:100]}...")
        return result

    def describe_environment(self) -> str:
        try:
            frame = self._capture_frame()
            image_b64 = self._frame_to_base64(frame)
            return self._ask_vision(
                image_b64,
                "Describe what you see in this image concisely. "
                "Focus on people, objects, the environment, and what's happening. "
                "Be specific and practical.",
            )
        except Exception as e:
            print(f"[Vision] Error in describe_environment: {e}")
            return f"Error describing environment: {e}"

    def read_text_in_frame(self) -> str:
        try:
            frame = self._capture_frame()
            image_b64 = self._frame_to_base64(frame)
            return self._ask_vision(
                image_b64,
                "Read and transcribe all text visible in this image. "
                "Preserve formatting if possible. If no text is present, say 'No text found'.",
            )
        except Exception as e:
            print(f"[Vision] Error in read_text_in_frame: {e}")
            return f"Error reading text: {e}"

    def analyze_image_file(self, path: str, prompt: Optional[str] = None) -> str:
        try:
            if not path or not path.strip():
                return "Error: No file path provided"
            print(f"[Vision] Analyzing image file: {path}")
            with open(path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            return self._ask_vision(image_b64, prompt or "Analyze this image in detail.")
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            print(f"[Vision] Error in analyze_image_file: {e}")
            return f"Error analyzing image: {e}"

    def capture_image(self) -> Optional[str]:
        """Capture a frame, save it, and return the file path."""
        try:
            frame = self._capture_frame()
            screenshots_dir = os.path.expanduser("~/jarvis_files/screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(screenshots_dir, f"{timestamp}.jpg")
            if cv2.imwrite(output_path, frame):
                print(f"[Vision] Image saved: {output_path}")
                return output_path
            return None
        except Exception as e:
            print(f"[Vision] Error in capture_image: {e}")
            return None

    def detect_motion(self, threshold: int = 500) -> bool:
        try:
            print(f"[Vision] Detecting motion (threshold: {threshold} pixels)...")
            cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}")
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            cap.release()
            if not ret1 or not ret2:
                raise RuntimeError("Could not read frames for motion detection")
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            motion_area: int = cv2.countNonZero(thresh)
            motion_detected = motion_area > threshold
            print(f"[Vision] Motion area: {motion_area}px — detected: {motion_detected}")
            return motion_detected
        except Exception as e:
            print(f"[Vision] Error in detect_motion: {e}")
            return False

    def take_screenshot(self, output_path: Optional[str] = None) -> str:
        try:
            frame = self._capture_frame()
            if not output_path:
                screenshots_dir = os.path.expanduser("~/jarvis_files/screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(screenshots_dir, f"{timestamp}.jpg")
            if cv2.imwrite(output_path, frame):
                print(f"[Vision] Screenshot saved: {output_path}")
                return output_path
            return f"Failed to save screenshot to {output_path}"
        except Exception as e:
            print(f"[Vision] Error in take_screenshot: {e}")
            return f"Screenshot error: {e}"
