"""
Tool Executor for J.A.R.V.I.S. Agent

Executes tool calls returned by the LLM.
All tools use free services—no API keys required (except optional Gmail/Calendar).
"""

import json
import subprocess
import os
import requests
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ddgs import DDGS
from typing import Any, Dict, Optional

load_dotenv()


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
        ha_token: Optional[str] = None,
        memory=None
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
            self.ha_url: Optional[str] = home_assistant_url or os.getenv("HOME_ASSISTANT_URL") or None
            self.ha_token: Optional[str] = ha_token or os.getenv("HOME_ASSISTANT_TOKEN") or None
            self.memory = memory  # injected from JarvisAgent
            camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))

            # Lazy-import vision to avoid hard dependency on cv2/ollama at startup
            self.vision = None
            try:
                from jarvis.vision.vision_module import VisionModule
                self.vision = VisionModule(camera_index=camera_index)
            except Exception as ve:
                print(f"[ToolExecutor] Vision module unavailable: {ve}")

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
                "system_settings": self._system_settings,
                "find_and_open": self._find_and_open,
                "system_volume": self._system_volume,
                "save_user_fact": self._save_user_fact,
                "get_user_facts": self._get_user_facts,
                "open_application": self._open_application,
                "spotify_control": self._spotify_control,
            }

            handler: Optional[Any] = handlers.get(tool_name)
            if not handler:
                error_msg: str = f"Unknown tool: '{tool_name}'"
                print(f"[ToolExecutor] {error_msg}")
                return f"Error: {error_msg}"

            # Execute the handler (args may be None for no-param tools)
            result: str = handler(**(args or {}))
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

        Returns text snippets ready for the LLM to synthesize — do NOT
        open any URLs from these results, just read the snippet text.
        """
        try:
            if not query:
                return "Error: Empty search query"

            print(f"[ToolExecutor] Web search: {query}")
            max_results = min(max_results, 8)

            with DDGS() as ddgs:
                # Try instant answer first (gives direct factual answers)
                try:
                    answers = list(ddgs.answers(query))
                    if answers:
                        answer_text = answers[0].get("text", "")
                        if answer_text:
                            return f"[Instant Answer] {answer_text}"
                except Exception:
                    pass

                results: list = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No search results found for: {query}"

            # Return only title + snippet — omit URLs to prevent the LLM
            # from trying to open them instead of reading the text.
            formatted: list = ["Search results (read these to answer — do not open URLs):"]
            for i, result in enumerate(results, 1):
                formatted.append(f"{i}. {result['title']}\n   {result['body']}")
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

    def _get_datetime(self, **_) -> str:
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
        Execute code or shell commands and return output.

        Detects whether the input is a shell command or Python code and
        runs it accordingly. Shell commands run via bash; Python code runs
        via the Python interpreter.

        Args:
            code: Shell command(s) or Python code to execute

        Returns:
            Code output (stdout + stderr) or error message
        """
        try:
            if not code:
                return "Error: No code provided"

            # Detect shell code. First try explicit markers, then fall back
            # to compile() — if Python rejects it with SyntaxError it must be
            # a shell command (e.g. "rm ~/file.txt", "mv a b", "git status").
            _SHELL_PREFIXES = (
                "#!", "sudo ", "echo ", "cat ", "ls ", "ll ", "rm ", "mv ",
                "cp ", "mkdir ", "touch ", "chmod ", "chown ", "grep ", "find ",
                "kill ", "killall ", "ps ", "top ", "df ", "du ", "tar ",
                "curl ", "wget ", "git ", "cd ", "export ", "source ", "unset ",
                "which ", "man ", "head ", "tail ", "sed ", "awk ", "sort ",
                "uniq ", "wc ", "xargs ", "dnf ", "apt ", "pip ", "pip3 ",
                "systemctl ", "service ", "modprobe ", "sensors", "ffmpeg ",
                "convert ", "xrandr ", "pactl ", "nmcli ", "bluetoothctl ",
                "dbus-send ", "flatpak ", "snap ", "xdg-open ",
            )
            stripped = code.strip()
            is_shell = (
                any(stripped.startswith(p) for p in _SHELL_PREFIXES)
                or " | " in code
                or " && " in code
                or " || " in code
                or " ; " in code
            )

            # Final fallback: if Python's compiler rejects it, treat as shell
            if not is_shell:
                try:
                    compile(code, "<string>", "exec")
                except SyntaxError:
                    is_shell = True

            if is_shell:
                print(f"[ToolExecutor] Running shell command...")
                # If the command needs sudo and SUDO_PASS is set, pipe it via sudo -S
                sudo_pass = os.getenv("SUDO_PASS", "")
                if sudo_pass and "sudo" in code:
                    # Replace bare `sudo` with `sudo -S` and pipe the password via stdin
                    patched = code.replace("sudo ", "sudo -S ", 1)
                    result = subprocess.run(
                        patched,
                        shell=True,
                        input=sudo_pass + "\n",
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                else:
                    result = subprocess.run(
                        code,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
            else:
                print(f"[ToolExecutor] Running Python code...")
                result = subprocess.run(
                    ["python3", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            output: str = result.stdout or result.stderr or "(no output)"
            return output[:2000]
        except subprocess.TimeoutExpired:
            return "Error: Execution timeout"
        except Exception as e:
            print(f"[ToolExecutor] Error in _run_code: {e}")
            return f"Code execution error: {e}"

    def _read_file(self, path: str, search_hint: str = "") -> str:
        """
        Read any text file on the system.

        Resolution order:
        1. Absolute path (/...) → used directly
        2. Home-relative (~/...) → expanded
        3. Bare relative → ~/jarvis_files/<path>
        4. Not found → search home dir for the filename;
           search_hint (e.g. "jarvis") narrows to paths containing that word.

        Args:
            path: File path or filename to locate
            search_hint: Optional keyword to prefer matching search results

        Returns:
            File contents (first 2000 chars), or candidate list, or error
        """
        try:
            if not path:
                return "Error: No file path provided"

            if path.startswith("/") or path.startswith("~"):
                full_path: str = os.path.expanduser(path)
            else:
                safe_base: str = os.path.expanduser("~/jarvis_files")
                full_path = os.path.join(safe_base, path.lstrip("/"))

            # ── Filesystem search fallback ────────────────────────────────
            if not os.path.exists(full_path):
                filename = os.path.basename(path) or path
                search_root = os.path.expanduser("~")
                print(f"[ToolExecutor] '{full_path}' not found — searching for '{filename}'")
                result = subprocess.run(
                    ["find", search_root, "-maxdepth", "8",
                     "-name", filename,
                     "-not", "-path", "*/node_modules/*",
                     "-not", "-path", "*/__pycache__/*"],
                    capture_output=True, text=True, timeout=10
                )
                matches = [l.strip() for l in result.stdout.splitlines() if l.strip()]

                if not matches:
                    return f"Error: '{filename}' not found anywhere under {search_root}."

                # Prefer matches that contain the search hint
                if search_hint:
                    preferred = [m for m in matches if search_hint.lower() in m.lower()]
                    if preferred:
                        matches = preferred

                if len(matches) == 1:
                    full_path = matches[0]
                    print(f"[ToolExecutor] Found: {full_path}")
                else:
                    listing = "\n".join(matches[:10])
                    return (
                        f"Found {len(matches)} files named '{filename}':\n{listing}\n"
                        f"Specify which one to read (use the full path)."
                    )
            # ─────────────────────────────────────────────────────────────

            with open(full_path, "r") as f:
                content: str = f.read()

            print(f"[ToolExecutor] Read file: {full_path}")
            return f"[File: {full_path}]\n\n{content[:2000]}"
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
        device: str = "",
        action: str = "",
        value: Optional[int] = None,
        **kwargs
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
            # Redirect misrouted system calls (8b model sometimes calls smart_home
            # for OS settings like power_mode, wifi, volume)
            power_actions = ("set_power_mode", "set_profile", "power_profile")
            if action in power_actions or kwargs.get("mode") or kwargs.get("profile"):
                mode = kwargs.get("mode") or kwargs.get("profile") or str(value or "")
                return self._system_settings(
                    setting="power_profile", action="set", value=mode
                )
            wifi_actions = ("enable_wifi", "disable_wifi", "wifi_on", "wifi_off")
            if action in wifi_actions:
                new_action = "on" if "on" in action or "enable" in action else "off"
                return self._system_settings(setting="wifi", action=new_action)

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

    def _capture_image(self, **_) -> str:
        """
        Capture an image from webcam.

        Returns:
            Base64-encoded JPEG or error message
        """
        try:
            if not self.vision:
                return "Error: Vision module not available (opencv not installed)"
            print(f"[ToolExecutor] Capturing image from webcam...")
            image_path: str = self.vision.capture_image()
            if image_path:
                return f"Image captured and saved to: {image_path}"
            else:
                return "Error: Failed to capture image"
        except Exception as e:
            print(f"[ToolExecutor] Error in _capture_image: {e}")
            return f"Image capture error: {e}"

    def _describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Describe an image using LLaVA vision AI.

        Args:
            image_path: Path to image or 'camera' for webcam
            prompt: Optional custom question about image

        Returns:
            Image description or error
        """
        try:
            if not self.vision:
                return "Error: Vision module not available (opencv not installed)"
            if not image_path:
                return "Error: No image path provided"

            print(f"[ToolExecutor] Analyzing image: {image_path}")

            if image_path.lower() == "camera":
                captured_path: Optional[str] = self.vision.capture_image()
                if not captured_path:
                    return "Error: Failed to capture image from camera"
                image_path = captured_path

            if not prompt:
                prompt = "Describe what you see in this image in detail."

            description: str = self.vision.analyze_image_file(image_path, prompt)
            return description
        except Exception as e:
            print(f"[ToolExecutor] Error in _describe_image: {e}")
            return f"Image analysis error: {e}"

    def _detect_motion(self, **_) -> str:
        """Detect motion on webcam."""
        try:
            if not self.vision:
                return "Error: Vision module not available (opencv not installed)"
            print(f"[ToolExecutor] Detecting motion...")
            motion_detected: bool = self.vision.detect_motion()
            return "Motion detected on camera" if motion_detected else "No motion detected"
        except Exception as e:
            print(f"[ToolExecutor] Error in _detect_motion: {e}")
            return f"Motion detection error: {e}"

    def _system_settings(self, setting: str, action: str, value: str = "", **_) -> str:
        """Control OS-level system settings via xrandr, nmcli, bluetoothctl, gsettings."""
        try:
            import re as _re
            # Normalize setting aliases
            setting = setting.lower().strip()
            if setting in ("theme", "color_scheme", "color-scheme", "appearance"):
                setting = "dark_mode"
            print(f"[ToolExecutor] System settings: {setting} → {action}")

            # ── Brightness (xrandr) ───────────────────────────────────────
            if setting == "brightness":
                if action == "get":
                    out = subprocess.run(
                        ["xrandr", "--verbose"], capture_output=True, text=True
                    ).stdout
                    m = _re.search(r"Brightness:\s*([\d.]+)", out)
                    pct = int(float(m.group(1)) * 100) if m else "unknown"
                    return f"Screen brightness is at {pct}%."
                # action="set" uses the value field; otherwise action IS the value
                raw = value if action == "set" else action
                try:
                    val = float(raw.rstrip("%")) / 100
                    val = max(0.1, min(1.0, val))
                except ValueError:
                    return "Provide a brightness value (e.g. '80' or '80%')."
                # Apply to all connected outputs
                outputs = _re.findall(r"^(\S+) connected", subprocess.run(
                    ["xrandr"], capture_output=True, text=True
                ).stdout, _re.MULTILINE)
                for output in outputs:
                    subprocess.run(
                        ["xrandr", "--output", output, "--brightness", str(val)],
                        capture_output=True
                    )
                return f"Screen brightness set to {int(val * 100)}%."

            # ── Wi-Fi (nmcli) ─────────────────────────────────────────────
            elif setting == "wifi":
                if action == "get":
                    out = subprocess.run(
                        ["nmcli", "-t", "-f", "WIFI", "radio"], capture_output=True, text=True
                    ).stdout.strip()
                    return f"Wi-Fi is {out}."
                elif action in ("on", "off"):
                    subprocess.run(["nmcli", "radio", "wifi", action], capture_output=True)
                    return f"Wi-Fi turned {action}."
                elif action == "toggle":
                    state = subprocess.run(
                        ["nmcli", "-t", "-f", "WIFI", "radio"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    new = "off" if state == "enabled" else "on"
                    subprocess.run(["nmcli", "radio", "wifi", new], capture_output=True)
                    return f"Wi-Fi toggled {new}."
                return f"Unknown wifi action '{action}'."

            # ── Bluetooth (bluetoothctl) ───────────────────────────────────
            elif setting == "bluetooth":
                if action == "get":
                    out = subprocess.run(
                        ["bluetoothctl", "show"], capture_output=True, text=True
                    ).stdout
                    powered = "on" if "Powered: yes" in out else "off"
                    return f"Bluetooth is {powered}."
                elif action in ("on", "off"):
                    subprocess.run(
                        ["bluetoothctl", "power", action], capture_output=True
                    )
                    return f"Bluetooth powered {action}."
                elif action == "toggle":
                    out = subprocess.run(
                        ["bluetoothctl", "show"], capture_output=True, text=True
                    ).stdout
                    new = "off" if "Powered: yes" in out else "on"
                    subprocess.run(["bluetoothctl", "power", new], capture_output=True)
                    return f"Bluetooth toggled {new}."
                return f"Unknown bluetooth action '{action}'."

            # ── Dark / Light Mode (gsettings) ─────────────────────────────
            elif setting == "dark_mode":
                if action == "get":
                    scheme = subprocess.run(
                        ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    mode = "dark" if "dark" in scheme else "light"
                    return f"Color scheme is currently {mode} mode."
                elif action in ("on", "dark"):
                    subprocess.run(["gsettings", "set", "org.gnome.desktop.interface",
                                    "color-scheme", "prefer-dark"], capture_output=True)
                    return "Dark mode enabled."
                elif action in ("off", "light"):
                    subprocess.run(["gsettings", "set", "org.gnome.desktop.interface",
                                    "color-scheme", "prefer-light"], capture_output=True)
                    return "Light mode enabled."
                elif action == "set":
                    # value field: "dark"/"on" → dark mode, "light"/"off" → light mode
                    v = value.lower().strip()
                    if v in ("dark", "on"):
                        subprocess.run(["gsettings", "set", "org.gnome.desktop.interface",
                                        "color-scheme", "prefer-dark"], capture_output=True)
                        return "Dark mode enabled."
                    elif v in ("light", "off"):
                        subprocess.run(["gsettings", "set", "org.gnome.desktop.interface",
                                        "color-scheme", "prefer-light"], capture_output=True)
                        return "Light mode enabled."
                    return f"Unknown dark_mode value '{value}'. Use 'dark' or 'light'."
                elif action == "toggle":
                    scheme = subprocess.run(
                        ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    new = "prefer-light" if "dark" in scheme else "prefer-dark"
                    subprocess.run(["gsettings", "set", "org.gnome.desktop.interface",
                                    "color-scheme", new], capture_output=True)
                    return f"Switched to {'dark' if 'dark' in new else 'light'} mode."
                return f"Unknown dark_mode action '{action}'."

            # ── Night Light (gsettings) ───────────────────────────────────
            elif setting == "night_light":
                key = "org.gnome.settings-daemon.plugins.color"
                if action == "get":
                    state = subprocess.run(
                        ["gsettings", "get", key, "night-light-enabled"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    return f"Night light is {'on' if state == 'true' else 'off'}."
                elif action in ("on", "enable"):
                    subprocess.run(["gsettings", "set", key, "night-light-enabled", "true"],
                                   capture_output=True)
                    return "Night light enabled."
                elif action in ("off", "disable"):
                    subprocess.run(["gsettings", "set", key, "night-light-enabled", "false"],
                                   capture_output=True)
                    return "Night light disabled."
                elif action == "toggle":
                    state = subprocess.run(
                        ["gsettings", "get", key, "night-light-enabled"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    new = "false" if state == "true" else "true"
                    subprocess.run(["gsettings", "set", key, "night-light-enabled", new],
                                   capture_output=True)
                    return f"Night light {'disabled' if new == 'false' else 'enabled'}."
                return f"Unknown night_light action '{action}'."

            # ── Power Profile (tuned-adm) ─────────────────────────────────
            elif setting == "power_profile":
                profile_map: Dict[str, str] = {
                    "balanced": "balanced",
                    "performance": "throughput-performance",
                    "power-saver": "powersave",
                    "powersave": "powersave",
                    "power saving": "powersave",
                    "battery": "balanced-battery",
                    "battery saving": "balanced-battery",
                    "battery-saving": "balanced-battery",
                    "latency": "latency-performance",
                    "desktop": "desktop",
                }
                if action == "get":
                    out = subprocess.run(
                        ["tuned-adm", "active"], capture_output=True, text=True
                    )
                    return f"Power profile: {out.stdout.strip()}."
                # action="set" uses value field; otherwise action IS the profile name
                raw_profile = value if action == "set" else action
                profile = profile_map.get(raw_profile.lower(), raw_profile)

                def _run_tuned(use_sudo: bool) -> subprocess.CompletedProcess:
                    cmd = ["tuned-adm", "profile", profile]
                    if use_sudo:
                        sudo_pass = os.getenv("SUDO_PASSWORD", "")
                        if sudo_pass:
                            return subprocess.run(
                                ["sudo", "-S"] + cmd,
                                input=sudo_pass + "\n",
                                capture_output=True, text=True
                            )
                        return subprocess.run(
                            ["sudo"] + cmd, capture_output=True, text=True
                        )
                    return subprocess.run(cmd, capture_output=True, text=True)

                # Try without sudo first (works if user has polkit rights)
                r = _run_tuned(use_sudo=False)
                if r.returncode == 0:
                    return f"Power profile set to '{profile}'."
                # Try with sudo (uses SUDO_PASSWORD from .env if set)
                r2 = _run_tuned(use_sudo=True)
                if r2.returncode == 0:
                    return f"Power profile set to '{profile}'."
                return f"Could not set power profile: {(r2.stderr or r.stderr).strip()}"

            return f"Unknown setting '{setting}'."
        except Exception as e:
            print(f"[ToolExecutor] Error in _system_settings: {e}")
            return f"System settings error: {e}"

    def _find_and_open(self, query: str, open_with: str, search_path: str = "") -> str:
        """Search filesystem for a directory/file and open it in the specified app."""
        try:
            import glob as _glob
            search_root = os.path.expanduser(search_path.strip() or "~")
            print(f"[ToolExecutor] Searching '{search_root}' for '{query}' → open with {open_with}")

            # Run find, limit depth to keep it fast
            result = subprocess.run(
                ["find", search_root, "-maxdepth", "6",
                 "-iname", f"*{query}*",
                 "-not", "-path", "*/.*",        # skip hidden dirs
                 "-not", "-path", "*/node_modules/*",
                 "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10
            )
            matches = [l.strip() for l in result.stdout.splitlines() if l.strip()]

            if not matches:
                return f"No files or directories matching '{query}' found under {search_root}."

            # Prefer exact name matches and directories first
            def rank(p: str) -> int:
                name = os.path.basename(p).lower()
                exact = name == query.lower()
                is_dir = os.path.isdir(p)
                return (0 if exact else 1) + (0 if is_dir else 2)

            matches.sort(key=rank)
            target = matches[0]

            # Resolve the app command
            app_map: Dict[str, list] = {
                "vscode": ["code", target],
                "code":   ["code", target],
                "dolphin": ["dolphin", target],
                "nautilus": ["nautilus", target],
                "files": ["nautilus", target],
                "terminal": ["gnome-terminal", f"--working-directory={target}"],
                "default": ["xdg-open", target],
            }
            cmd = app_map.get(open_with.lower())
            if not cmd:
                # Treat open_with as a raw binary
                cmd = [open_with, target]

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            others = len(matches) - 1
            note = f" ({others} other match{'es' if others != 1 else ''} found)" if others else ""
            return f"Opened '{target}' in {open_with}{note}."
        except subprocess.TimeoutExpired:
            return f"Search timed out. Try a more specific query or narrower search_path."
        except Exception as e:
            print(f"[ToolExecutor] Error in _find_and_open: {e}")
            return f"Find and open error: {e}"

    def _system_volume(self, action: str, level: Optional[int] = None) -> str:
        """Control OS-level system audio volume via pactl."""
        try:
            import re as _re
            print(f"[ToolExecutor] System volume: {action} {level or ''}")

            if action == "set":
                if level is None:
                    return "Error: provide a volume level (0-100)."
                level = max(0, min(100, level))
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                    check=True, capture_output=True
                )
                return f"System volume set to {level}%."

            elif action == "get":
                out = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True, text=True
                ).stdout
                m = _re.search(r"(\d+)%", out)
                pct = m.group(1) if m else "unknown"
                return f"System volume is at {pct}%."

            elif action == "mute":
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"],
                    check=True, capture_output=True
                )
                return "System audio muted."

            elif action == "unmute":
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                    check=True, capture_output=True
                )
                return "System audio unmuted."

            else:
                return f"Unknown action '{action}'."
        except Exception as e:
            print(f"[ToolExecutor] Error in _system_volume: {e}")
            return f"System volume error: {e}"

    def _save_user_fact(self, key: str, value: str) -> str:
        try:
            if not self.memory:
                return "Memory system unavailable."
            self.memory.set_fact(key, value)
            return f"Noted: {key} = {value}"
        except Exception as e:
            return f"Failed to save fact: {e}"

    def _get_user_facts(self, **_) -> str:
        try:
            if not self.memory:
                return "Memory system unavailable."
            facts = self.memory.get_all_facts()
            if not facts:
                return "No facts stored about you yet."
            return "\n".join(f"{k}: {v}" for k, v in facts.items())
        except Exception as e:
            return f"Failed to retrieve facts: {e}"

    def _open_application(self, app: str, args: str = "", **_) -> str:
        """
        Launch any installed application.

        Tries (in order): known alias → PATH binary → flatpak → snap → xdg-open.

        Args:
            app: Application name (e.g. 'firefox', 'spotify', 'vlc')
            args: Optional extra arguments/URL to pass

        Returns:
            Launch status message
        """
        try:
            if not app:
                return "Error: No application specified"

            print(f"[ToolExecutor] Opening application: {app} {args}".strip())

            # Terminal emulators — used to detect when to pass commands via bash -c
            _TERMINAL_CMDS = {
                "gnome-terminal", "xterm", "konsole", "alacritty",
                "kitty", "tilix", "xfce4-terminal", "foot",
            }

            # Common name → command aliases
            aliases: Dict[str, str] = {
                "spotify": "flatpak run com.spotify.Client",
                "discord": "flatpak run com.discordapp.Discord",
                "chrome": "google-chrome",
                "google chrome": "google-chrome",
                "chromium": "chromium-browser",
                "brave": "brave-browser",
                "firefox": "firefox",
                "browser": "__default_browser__",
                "terminal": "gnome-terminal",
                "files": "nautilus",
                "explorer": "nautilus",
                "vscode": "code",
                "vs code": "code",
                "editor": "gedit",
                "calculator": "gnome-calculator",
                "steam": "flatpak run com.valvesoftware.Steam",
                "obs": "flatpak run com.obsproject.Studio",
                "vlc": "vlc",
                "gimp": "gimp",
                "inkscape": "inkscape",
            }

            # Map .desktop file names → browser commands (for xdg-settings output)
            desktop_to_cmd: Dict[str, str] = {
                "google-chrome.desktop": "google-chrome",
                "chromium-browser.desktop": "chromium-browser",
                "chromium.desktop": "chromium",
                "brave-browser.desktop": "brave-browser",
                "firefox.desktop": "firefox",
                "org.mozilla.firefox.desktop": "firefox",
                "microsoft-edge.desktop": "microsoft-edge",
            }

            def _default_browser() -> str:
                """Return the system default browser command, falling back to chrome."""
                try:
                    desktop = subprocess.run(
                        ["xdg-settings", "get", "default-web-browser"],
                        capture_output=True, text=True, timeout=3
                    ).stdout.strip()
                    cmd = desktop_to_cmd.get(desktop)
                    if cmd:
                        return cmd
                except Exception:
                    pass
                # Fallback: first installed browser in priority order
                for candidate in ("google-chrome", "firefox", "chromium-browser", "brave-browser"):
                    try:
                        r = subprocess.run(["which", candidate], capture_output=True, timeout=2)
                        if r.returncode == 0:
                            return candidate
                    except Exception:
                        pass
                return "xdg-open"

            # Incognito / private-window: detect keyword in app or args
            app_lower = app.lower()
            incognito = (
                "incognito" in app_lower or "private" in app_lower
                or "incognito" in args.lower() or "private" in args.lower()
            )
            # Strip incognito keywords from args
            clean_args = (
                args
                .replace("--incognito", "").replace("incognito", "")
                .replace("--private-window", "").replace("private", "")
                .replace("mode", "").strip()
            )

            # Resolve app name (strip incognito keywords first)
            base_app = app_lower.replace("incognito", "").replace("private", "").strip()
            resolved = aliases.get(base_app, base_app or app)

            # Resolve __default_browser__ placeholder or generic browser requests
            if resolved == "__default_browser__" or (
                incognito and base_app in ("", "browser", "web browser", "the browser")
            ):
                resolved = _default_browser()

            # Choose the correct incognito flag for the browser
            if incognito:
                incognito_flag = "--private-window" if "firefox" in resolved else "--incognito"
                args_parts = [incognito_flag] + ([clean_args] if clean_args else [])
            else:
                args_parts = clean_args.split() if clean_args else (args.split() if args else [])

            cmd_parts = resolved.split() + args_parts

            # If opening a terminal with shell commands, run them inside it
            # so the user sees the output in the terminal window.
            resolved_bin = resolved.split()[0]
            if resolved_bin in _TERMINAL_CMDS and clean_args and not clean_args.startswith("--"):
                cmd_parts = [resolved_bin, "--", "bash", "-c", f"{clean_args}; exec bash"]

            # 1. Try the resolved/aliased command directly
            try:
                subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return f"Launched {app}."
            except FileNotFoundError:
                pass

            # 2. Try flatpak (search by partial name)
            try:
                fp_list = subprocess.run(
                    ["flatpak", "list", "--app", "--columns=application"],
                    capture_output=True, text=True, timeout=5
                )
                matches = [
                    line for line in fp_list.stdout.splitlines()
                    if app.lower() in line.lower()
                ]
                if matches:
                    fp_id = matches[0].strip()
                    launch_cmd = ["flatpak", "run", fp_id]
                    if args:
                        launch_cmd.append(args)
                    subprocess.Popen(
                        launch_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return f"Launched {app} (Flatpak: {fp_id})."
            except Exception:
                pass

            # 3. Try snap
            try:
                subprocess.Popen(
                    ["snap", "run", app] + ([args] if args else []),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return f"Launched {app} via Snap."
            except Exception:
                pass

            # 4. xdg-open fallback
            subprocess.Popen(
                ["xdg-open", args or app],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return f"Attempted to open {app} via system default handler."

        except Exception as e:
            print(f"[ToolExecutor] Error in _open_application: {e}")
            return f"Could not open {app}: {e}"

    def _spotify_control(
        self,
        action: str,
        query: str = "",
        value: Optional[int] = None
    ) -> str:
        """
        Control Spotify via MPRIS2 over DBus.

        Args:
            action: play, pause, play_pause, next, previous, stop, search, open, status, volume
            query: Search string (for 'search' action)
            value: Volume 0-100 (for 'volume' action)

        Returns:
            Result message
        """
        DEST = "org.mpris.MediaPlayer2.spotify"
        PATH = "/org/mpris/MediaPlayer2"
        PLAYER = "org.mpris.MediaPlayer2.Player"

        def mpris(method: str) -> bool:
            result = subprocess.run(
                ["dbus-send", "--session", "--type=method_call",
                 f"--dest={DEST}", PATH, f"{PLAYER}.{method}"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0

        def now_playing() -> str:
            """Fetch the current track from MPRIS and return a formatted string."""
            import re as _re
            import time as _time
            _time.sleep(0.6)  # wait for player to update after track change
            result = subprocess.run(
                ["dbus-send", "--session", "--print-reply",
                 f"--dest={DEST}", PATH,
                 "org.freedesktop.DBus.Properties.GetAll",
                 f"string:{PLAYER}"],
                capture_output=True, text=True, timeout=5
            )
            out = result.stdout
            if not out:
                return ""

            def extract(key: str) -> str:
                m = _re.search(rf'string "{key}"\s+variant\s+string "([^"]+)"', out)
                return m.group(1) if m else "unknown"

            title = extract("xesam:title")
            artist_m = _re.search(
                r'string "xesam:artist".*?string "([^"]+)"', out, _re.DOTALL
            )
            artist = artist_m.group(1) if artist_m else extract("xesam:artist")
            if title == "unknown":
                return ""
            return f"Now playing: \"{title}\" by {artist}."

        try:
            print(f"[ToolExecutor] Spotify: {action} {query or ''}")

            if action == "play":
                mpris("Play")
                track = now_playing()
                return f"Spotify: playing. {track}".strip()

            elif action == "pause":
                mpris("Pause")
                return "Spotify: paused."

            elif action == "play_pause":
                mpris("PlayPause")
                track = now_playing()
                return f"Spotify: toggled. {track}".strip()

            elif action == "next":
                mpris("Next")
                track = now_playing()
                return f"Spotify: skipped. {track}" if track else "Spotify: skipped to next track."

            elif action == "previous":
                mpris("Previous")
                track = now_playing()
                return f"Spotify: went back. {track}" if track else "Spotify: went to previous track."

            elif action == "stop":
                mpris("Stop")
                return "Spotify: stopped."

            elif action == "open":
                subprocess.Popen(
                    ["flatpak", "run", "com.spotify.Client"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return "Spotify: opening."

            elif action == "search":
                if not query:
                    return "Error: Provide a search query."
                encoded = query.replace(" ", "%20")
                subprocess.Popen(
                    ["xdg-open", f"spotify:search:{encoded}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return f"Spotify: searching for '{query}'."

            elif action == "volume":
                if value is None:
                    return "Error: Provide a volume level (0-100)."
                vol = max(0.0, min(1.0, value / 100.0))
                subprocess.run(
                    ["dbus-send", "--session", "--type=method_call",
                     f"--dest={DEST}", PATH,
                     "org.freedesktop.DBus.Properties.Set",
                     f"string:{PLAYER}", "string:Volume",
                     f"variant:double:{vol}"],
                    capture_output=True, text=True, timeout=5
                )
                return f"Spotify: volume set to {value}%."

            elif action == "status":
                import re as _re
                result = subprocess.run(
                    ["dbus-send", "--session", "--print-reply",
                     f"--dest={DEST}", PATH,
                     "org.freedesktop.DBus.Properties.GetAll",
                     f"string:{PLAYER}"],
                    capture_output=True, text=True, timeout=5
                )
                out = result.stdout

                def extract(key: str) -> str:
                    m = _re.search(rf'string "{key}"\s+variant\s+string "([^"]+)"', out)
                    return m.group(1) if m else "unknown"

                status = extract("PlaybackStatus")
                title = extract("xesam:title")
                artist_m = _re.search(
                    r'string "xesam:artist".*?string "([^"]+)"', out, _re.DOTALL
                )
                artist = artist_m.group(1) if artist_m else "unknown"
                album = extract("xesam:album")
                return (
                    f"Spotify status: {status}\n"
                    f"Track: {title}\nArtist: {artist}\nAlbum: {album}"
                )

            else:
                return f"Error: Unknown Spotify action '{action}'."

        except FileNotFoundError:
            return "Error: dbus-send not found. Make sure dbus-tools is installed."
        except Exception as e:
            print(f"[ToolExecutor] Error in _spotify_control: {e}")
            return f"Spotify control error: {e}"
