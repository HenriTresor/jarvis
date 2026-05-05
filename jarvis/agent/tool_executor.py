"""
Tool Executor for J.A.R.V.I.S. Agent

Executes tool calls returned by the LLM.
All tools use free services—no API keys required (except optional Gmail/Calendar).
"""

import json
import subprocess
import os
import requests
from datetime import datetime
from duckduckgo_search import DDGS
from typing import Any, Dict, Optional
import base64
from pathlib import Path


class ToolExecutor:
    """
    Executes tool calls from the LLM.

    Routes tool names and arguments to their implementations.
    All tools are wrapped in try/except to prevent crashes.
    Returns safe string fallbacks on any error.

    Example:
        executor = ToolExecutor(
            home_assistant_url="http://localhost:8123",
            ha_token="your_token"
        )
        result = executor.execute("web_search", {"query": "python 3.12"})
        print(result)  # Search results as string
    """

    def __init__(
        self,
        home_assistant_url: Optional[str] = None,
        ha_token: Optional[str] = None
    ) -> None:
        """
        Initialize the tool executor.

        Args:
            home_assistant_url: Home Assistant base URL (e.g., "http://localhost:8123")
            ha_token: Long-lived Home Assistant token

        Returns:
            None
        """
        try:
            self.ha_url: Optional[str] = home_assistant_url
            self.ha_token: Optional[str] = ha_token

            if self.ha_url:
                print(f"[ToolExecutor] Home Assistant configured: {self.ha_url}")
            else:
                print(f"[ToolExecutor] Home Assistant not configured (smart home will be simulated)")

            print(f"[ToolExecutor] Tool executor initialized.")
        except Exception as e:
            print(f"[ToolExecutor] Error in __init__: {e}")

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool by name with the given arguments.

        Routes to the appropriate handler and returns the result as a string.
        All errors are caught and returned as error messages.

        Args:
            tool_name: Name of the tool to execute (e.g., 'web_search')
            args: Dict of tool arguments (e.g., {'query': 'python'})

        Returns:
            String result from the tool, or error message if execution fails

        Raises:
            Exception: On execution error (caught internally, returns error string)
        """
        try:
            # Map tool names to handler methods
            handlers: Dict[str, Any] = {
                "web_search": self._web_search,
                "get_weather": self._get_weather,
                "get_datetime": self._get_datetime,
                "run_code": self._run_code,
                "read_file": self._read_file,
                "write_file": self._write_file,
                "smart_home_control": self._smart_home_control,
                "save_note": self._save_note,
                "send_email": self._send_email,
                "get_unread_emails": self._get_unread_emails,
                "get_upcoming_events": self._get_upcoming_events,
                "create_calendar_event": self._create_calendar_event,
                "capture_image": self._capture_image,
                "describe_image": self._describe_image,
                "detect_motion": self._detect_motion,
            }

            handler: Optional[Any] = handlers.get(tool_name)
            if not handler:
                error_msg: str = f"Unknown tool: '{tool_name}'"
                print(f"[ToolExecutor] {error_msg}")
                return f"Error: {error_msg}"

            # Execute the handler
            result: str = handler(**args)
            print(f"[ToolExecutor] {tool_name} executed successfully")
            return result
        except Exception as e:
            print(f"[ToolExecutor] Error in execute: {e}")
            return f"Tool execution error: {e}"

    # ─────────────────────────────────────────────────────────────────────
    # Tool Implementations
    # ─────────────────────────────────────────────────────────────────────

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """
        Search the web using DuckDuckGo (completely free, no API key).

        Args:
            query: Search query string
            max_results: Number of results to return (1-10)

        Returns:
            Formatted search results as string
        """
        try:
            if not query:
                return "Error: Empty search query"

            print(f"[ToolExecutor] Web search: {query}")
            max_results = min(max_results, 10)

            with DDGS() as ddgs:
                results: list = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No search results found for: {query}"

            formatted: list = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. {result['title']}\n{result['body']}\n{result['href']}\n"
                )
            return "\n".join(formatted)
        except Exception as e:
            print(f"[ToolExecutor] Error in _web_search: {e}")
            return f"Search error: {e}"

    def _get_weather(self, city: str) -> str:
        """
        Get weather for a city using Open-Meteo (free, no API key).

        Args:
            city: City name (e.g., 'Kigali', 'London')

        Returns:
            Formatted weather information
        """
        try:
            if not city:
                return "Error: No city specified"

            print(f"[ToolExecutor] Weather for: {city}")

            # Step 1: Geocode the city
            geo_url: str = (
                f"https://geocoding-api.open-meteo.com/v1/search"
                f"?name={city}&count=1&language=en"
            )
            geo_response = requests.get(geo_url, timeout=5)
            geo_data: Dict = geo_response.json()

            if not geo_data.get("results"):
                return f"Could not find weather data for '{city}'"

            location: Dict = geo_data["results"][0]
            lat: float = location["latitude"]
            lon: float = location["longitude"]
            name: str = location.get("name", city)

            # Step 2: Get weather
            weather_url: str = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m"
                f"&temperature_unit=celsius"
            )
            weather_response = requests.get(weather_url, timeout=5)
            weather_data: Dict = weather_response.json()
            current: Dict = weather_data["current"]

            # Map WMO weather codes to descriptions
            code_map: Dict[int, str] = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
                3: "Overcast", 45: "Foggy", 48: "Foggy", 51: "Light drizzle",
                53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
                63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow",
                73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
                80: "Slight rain showers", 81: "Moderate rain showers",
                82: "Violent rain showers", 85: "Slight snow showers",
                86: "Heavy snow showers", 95: "Thunderstorm",
                96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
            }
            condition: str = code_map.get(current["weather_code"], "Unknown")

            result: str = (
                f"Weather in {name}:\n"
                f"  Condition: {condition}\n"
                f"  Temperature: {current['temperature_2m']}°C\n"
                f"  Humidity: {current['relative_humidity_2m']}%\n"
                f"  Wind: {current['wind_speed_10m']} km/h"
            )
            return result
        except Exception as e:
            print(f"[ToolExecutor] Error in _get_weather: {e}")
            return f"Weather error: {e}"

    def _get_datetime(self) -> str:
        """
        Get current date and time.

        Returns:
            Formatted date/time string
        """
        try:
            now: str = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
            print(f"[ToolExecutor] DateTime: {now}")
            return now
        except Exception as e:
            print(f"[ToolExecutor] Error in _get_datetime: {e}")
            return f"DateTime error: {e}"

    def _run_code(self, code: str) -> str:
        """
        Execute Python code and return output.

        WARNING: This is a simple sandbox using subprocess.
        For production, use E2B or Docker sandbox.

        Args:
            code: Python code to execute

        Returns:
            Code output (stdout + stderr) or error message
        """
        try:
            if not code:
                return "Error: No code provided"

            print(f"[ToolExecutor] Running Python code...")

            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=10
            )
            output: str = result.stdout or result.stderr or "(no output)"
            return output[:1000]  # Truncate to 1000 chars
        except subprocess.TimeoutExpired:
            return "Error: Code execution timeout (>10 seconds)"
        except Exception as e:
            print(f"[ToolExecutor] Error in _run_code: {e}")
            return f"Code execution error: {e}"

    def _read_file(self, path: str) -> str:
        """
        Read a text file from ~/jarvis_files.

        Args:
            path: Relative file path

        Returns:
            File contents (first 2000 chars) or error message
        """
        try:
            if not path:
                return "Error: No file path provided"

            safe_base: str = os.path.expanduser("~/jarvis_files")
            full_path: str = os.path.join(safe_base, path.lstrip("/"))

            if not os.path.exists(full_path):
                return f"Error: File not found: {path}"

            with open(full_path, "r") as f:
                content: str = f.read()

            print(f"[ToolExecutor] Read file: {path}")
            return content[:2000]
        except Exception as e:
            print(f"[ToolExecutor] Error in _read_file: {e}")
            return f"File read error: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """
        Write content to a file in ~/jarvis_files.

        Creates directories if needed.

        Args:
            path: Relative file path
            content: Content to write

        Returns:
            Success message or error
        """
        try:
            if not path:
                return "Error: No file path provided"

            safe_base: str = os.path.expanduser("~/jarvis_files")
            full_path: str = os.path.join(safe_base, path.lstrip("/"))

            # Create directory if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w") as f:
                f.write(content)

            print(f"[ToolExecutor] Wrote file: {path}")
            return f"File written successfully: {path}"
        except Exception as e:
            print(f"[ToolExecutor] Error in _write_file: {e}")
            return f"File write error: {e}"

    def _smart_home_control(
        self,
        device: str,
        action: str,
        value: Optional[int] = None
    ) -> str:
        """
        Control smart home devices via Home Assistant.

        Args:
            device: Device name (e.g., 'desk lamp')
            action: Action (on, off, toggle, set_brightness)
            value: Brightness value 0-255 (for set_brightness)

        Returns:
            Result message
        """
        try:
            if not device or not action:
                return "Error: Missing device or action"

            print(f"[ToolExecutor] Smart home: {device} {action}")

            if not self.ha_url or not self.ha_token:
                # Simulate the action
                result: str = f"[SIMULATED] {device} turned {action}"
                if action == "set_brightness" and value:
                    result = f"[SIMULATED] {device} brightness set to {value}"
                return result

            # Send request to Home Assistant
            headers: Dict[str, str] = {
                "Authorization": f"Bearer {self.ha_token}"
            }

            service_map: Dict[str, str] = {
                "on": "turn_on",
                "off": "turn_off",
                "toggle": "toggle",
                "set_brightness": "turn_on"
            }
            service: str = service_map.get(action, "turn_on")

            entity_id: str = f"light.{device.replace(' ', '_').lower()}"
            data: Dict[str, Any] = {"entity_id": entity_id}
            if action == "set_brightness" and value is not None:
                data["brightness"] = value

            response = requests.post(
                f"{self.ha_url}/api/services/light/{service}",
                json=data,
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                return f"Device '{device}' {action} successfully"
            else:
                return f"Error: Home Assistant returned {response.status_code}"
        except Exception as e:
            print(f"[ToolExecutor] Error in _smart_home_control: {e}")
            return f"Smart home error: {e}"

    def _save_note(self, title: str, content: str) -> str:
        """
        Save a note to ~/jarvis_notes with timestamp.

        Args:
            title: Note title
            content: Note content

        Returns:
            Success message with filename
        """
        try:
            if not title:
                return "Error: No note title"

            notes_dir: str = os.path.expanduser("~/jarvis_notes")
            os.makedirs(notes_dir, exist_ok=True)

            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename: str = f"{timestamp}_{title.replace(' ', '_')}.txt"
            filepath: str = os.path.join(notes_dir, filename)

            with open(filepath, "w") as f:
                f.write(f"# {title}\n")
                f.write(f"Date: {datetime.now().isoformat()}\n\n")
                f.write(content)

            print(f"[ToolExecutor] Note saved: {filename}")
            return f"Note saved: {filename}"
        except Exception as e:
            print(f"[ToolExecutor] Error in _save_note: {e}")
            return f"Note save error: {e}"

    def _send_email(self, to: str, subject: str, body: str) -> str:
        """
        Send an email via Gmail (requires OAuth setup).

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body

        Returns:
            Success message or error
        """
        try:
            if not to or not subject:
                return "Error: Missing recipient or subject"

            print(f"[ToolExecutor] Sending email to: {to}")
            # Gmail integration would go here
            # For now, return simulated response
            return f"[SIMULATED] Email sent to {to}"
        except Exception as e:
            print(f"[ToolExecutor] Error in _send_email: {e}")
            return f"Email send error: {e}"

    def _get_unread_emails(self, max_results: int = 5) -> str:
        """
        Fetch unread emails from Gmail (requires OAuth setup).

        Args:
            max_results: Max emails to fetch

        Returns:
            Formatted email list or error
        """
        try:
            print(f"[ToolExecutor] Fetching unread emails...")
            # Gmail integration would go here
            # For now, return simulated response
            return "[SIMULATED] You have 3 unread emails"
        except Exception as e:
            print(f"[ToolExecutor] Error in _get_unread_emails: {e}")
            return f"Email fetch error: {e}"

    def _get_upcoming_events(self, max_events: int = 5) -> str:
        """
        Fetch upcoming calendar events (requires OAuth setup).

        Args:
            max_events: Max events to fetch

        Returns:
            Formatted event list or error
        """
        try:
            print(f"[ToolExecutor] Fetching upcoming events...")
            # Google Calendar integration would go here
            # For now, return simulated response
            return "[SIMULATED] You have 2 upcoming events today"
        except Exception as e:
            print(f"[ToolExecutor] Error in _get_upcoming_events: {e}")
            return f"Event fetch error: {e}"

    def _create_calendar_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = ""
    ) -> str:
        """
        Create a calendar event (requires OAuth setup).

        Args:
            title: Event title
            start: ISO 8601 start time
            end: ISO 8601 end time
            description: Optional description

        Returns:
            Success message or error
        """
        try:
            if not title or not start or not end:
                return "Error: Missing required fields (title, start, end)"

            print(f"[ToolExecutor] Creating event: {title}")
            # Google Calendar integration would go here
            # For now, return simulated response
            return f"[SIMULATED] Event '{title}' created for {start}"
        except Exception as e:
            print(f"[ToolExecutor] Error in _create_calendar_event: {e}")
            return f"Calendar event error: {e}"

    def _capture_image(self) -> str:
        """
        Capture an image from webcam.

        Returns:
            Base64-encoded JPEG or error message
        """
        try:
            print(f"[ToolExecutor] Capturing image from webcam...")
            # Vision integration would go here
            # For now, return simulated response
            return "[SIMULATED] Image captured and saved"
        except Exception as e:
            print(f"[ToolExecutor] Error in _capture_image: {e}")
            return f"Image capture error: {e}"

    def _describe_image(self, image_path: str, prompt: str = None) -> str:
        """
        Describe an image using LLaVA vision AI.

        Args:
            image_path: Path to image or 'camera' for webcam
            prompt: Optional custom question about image

        Returns:
            Image description or error
        """
        try:
            if not image_path:
                return "Error: No image path provided"

            print(f"[ToolExecutor] Analyzing image: {image_path}")
            # Vision integration would go here
            # For now, return simulated response
            return "[SIMULATED] Image contains a desk with a computer"
        except Exception as e:
            print(f"[ToolExecutor] Error in _describe_image: {e}")
            return f"Image analysis error: {e}"

    def _detect_motion(self) -> str:
        """
        Detect motion on webcam.

        Returns:
            Motion detection result
        """
        try:
            print(f"[ToolExecutor] Detecting motion...")
            # Vision integration would go here
            # For now, return simulated response
            return "[SIMULATED] No motion detected"
        except Exception as e:
            print(f"[ToolExecutor] Error in _detect_motion: {e}")
            return f"Motion detection error: {e}"
