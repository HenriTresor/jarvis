"""
Tool Schemas and Definitions for J.A.R.V.I.S. Agent

Defines all available tools using the OpenAI/Ollama function calling schema.
These schemas tell the LLM what tools are available and how to call them.

All tools are free and require no API keys:
- Web search: DuckDuckGo (free, no key)
- Weather: Open-Meteo (free, no key)
- Files: Local filesystem
- Smart home: Home Assistant (self-hosted)
- Code execution: Python subprocess (sandboxed)
"""

from typing import List, Dict, Any


# Complete tool schema list in OpenAI function calling format
# Compatible with both Ollama and Anthropic Claude APIs
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, facts, prices, schedules, "
                "scores, weather, people, products, or anything that may have changed since "
                "training. Write queries short and natural, the way a human would Google them. "
                "Do not append today's date to the query unless the topic is date-specific."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short natural search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather information for a city. "
                "Returns temperature, conditions, humidity, and wind speed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'Kigali', 'London', 'Tokyo')"
                    }
                },
                "required": ["city"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": (
                "Get the current date and time. "
                "Use this to answer 'what time is it?' or when scheduling events."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute shell commands or Python code on the user's system and return the output. "
                "Accepts bash shell commands (sudo, dnf, apt, systemctl, echo, sensors, etc.) "
                "OR Python code — auto-detected. Use for installations, system diagnostics, "
                "file operations, calculations, and any terminal command the user asks to run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Shell command(s) or Python code to execute. "
                            "Examples: 'sudo dnf install -y lm_sensors', "
                            "'sensors', 'echo 255 | sudo tee /sys/class/hwmon/hwmon0/pwm1'"
                        )
                    }
                },
                "required": ["code"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of any file anywhere on the system. "
                "Accepts absolute paths, home-relative paths (~/...), or just a filename. "
                "If the exact path is not found, searches the home directory automatically. "
                "Use search_hint to narrow results when a filename is ambiguous "
                "(e.g. path='.env', search_hint='jarvis' finds ~/jarvis/.env). "
                "Returns up to 2000 characters of the file content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path or filename. Examples: "
                            "'/etc/hosts', '~/jarvis/.env', '.env', 'config.json', "
                            "'~/Documents/resume.pdf'"
                        )
                    },
                    "search_hint": {
                        "type": "string",
                        "description": (
                            "Optional keyword to prefer matching search results when "
                            "multiple files share the same name. "
                            "E.g. 'jarvis' to prefer files inside a jarvis directory."
                        )
                    }
                },
                "required": ["path"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or create a text file on the filesystem. "
                "Files are stored in ~/jarvis_files. "
                "Creates directories if needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (e.g., 'documents/new_file.txt')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "smart_home_control",
            "description": (
                "Control smart home devices through Home Assistant. "
                "Supports lights, plugs, and switches. "
                "Examples: turn on desk lamp, set brightness to 50%, toggle living room lights"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device name (e.g., 'desk lamp', 'living room light')"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "toggle", "set_brightness"],
                        "description": (
                            "Action to perform: "
                            "'on' = turn on, "
                            "'off' = turn off, "
                            "'toggle' = switch state, "
                            "'set_brightness' = set brightness (0-255)"
                        )
                    },
                    "value": {
                        "type": "integer",
                        "description": (
                            "Brightness value 0-255 (only for set_brightness action). "
                            "0 = off, 255 = full brightness"
                        )
                    }
                },
                "required": ["device", "action"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": (
                "Save a note or reminder to the local notes folder. "
                "Creates a timestamped file. "
                "Useful for capturing thoughts and tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title (e.g., 'Project Ideas', 'Meeting Summary')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content/body text"
                    }
                },
                "required": ["title", "content"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email through Gmail. "
                "Requires Gmail API credentials to be configured. "
                "Always confirm recipient and content before sending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address (e.g., 'alice@example.com')"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_unread_emails",
            "description": (
                "Fetch unread emails from your Gmail inbox. "
                "Shows sender, subject, and preview. "
                "Requires Gmail API credentials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to fetch (default: 5)",
                        "default": 5
                    }
                }
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": (
                "Fetch upcoming calendar events from Google Calendar. "
                "Shows event title, time, and description. "
                "Requires Google Calendar API credentials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_events": {
                        "type": "integer",
                        "description": "Maximum number of events to fetch (default: 5)",
                        "default": 5
                    }
                }
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a new event on Google Calendar. "
                "Provide title, start time, end time, and optional description. "
                "Times should be in ISO 8601 format (e.g., '2025-05-05T14:00:00'). "
                "Requires Google Calendar API credentials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Event title (e.g., 'Team Meeting')"
                    },
                    "start": {
                        "type": "string",
                        "description": "Start time in ISO 8601 format (e.g., '2025-05-05T14:00:00')"
                    },
                    "end": {
                        "type": "string",
                        "description": "End time in ISO 8601 format (e.g., '2025-05-05T15:00:00')"
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description (optional)"
                    }
                },
                "required": ["title", "start", "end"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": (
                "Set a real alarm that fires a desktop notification and plays a sound "
                "at the specified time. Works even if this chat window is closed. "
                "Use this for 'wake me up at 7', 'alarm in 30 minutes', "
                "'remind me at 8 PM', 'set a timer for X'. "
                "Prefer this over create_calendar_event for time-sensitive reminders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": (
                            "When to trigger the alarm. Accepts: "
                            "'HH:MM' (24h, e.g. '07:30'), "
                            "'HH:MM AM/PM' (e.g. '8:00 PM'), "
                            "or full ISO datetime (e.g. '2026-05-18T08:00:00')."
                        )
                    },
                    "label": {
                        "type": "string",
                        "description": "What the alarm is for (e.g. 'Wake up', 'Take medicine', 'Meeting')",
                        "default": "Alarm"
                    }
                },
                "required": ["time"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": (
                "Capture an image from the webcam. "
                "Returns a base64-encoded JPEG image. "
                "Useful for taking screenshots or analyzing the environment."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": (
                "Analyze an image using vision AI (LLaVA). "
                "Describes objects, text, people, and scene. "
                "Can read text in images (OCR-like functionality)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file or 'camera' to capture from webcam"
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Custom question about the image. "
                            "Examples: 'What text is visible?', 'How many people?'"
                        )
                    }
                },
                "required": ["image_path"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "system_settings",
            "description": (
                "Control OS-level system settings: screen brightness, Wi-Fi, "
                "Bluetooth, dark/light mode, night light, and power profile. "
                "Use this for requests like 'turn on dark mode', 'disable WiFi', "
                "'set brightness to 80%', 'enable night light', 'check bluetooth'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "setting": {
                        "type": "string",
                        "enum": ["brightness", "wifi", "bluetooth", "dark_mode", "night_light", "power_profile"],
                        "description": "Which setting to control"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["get", "on", "off", "toggle", "set"],
                        "description": "'get'=read current, 'on'/'off'/'toggle' for switches, 'set'=apply a value (requires 'value' field)"
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Value for 'set' action. "
                            "Brightness: '0'–'100'. "
                            "Power profile: 'balanced', 'performance', 'battery-saving', 'power-saver'. "
                            "Dark mode: 'dark' or 'light'."
                        )
                    }
                },
                "required": ["setting", "action"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "find_and_open",
            "description": (
                "Search the filesystem for a directory or file by name, then open it "
                "in the application the user specifies. "
                "Use for: 'find my jarvis project and open in VSCode', "
                "'search for Downloads folder and open in Dolphin', "
                "'find the music folder and open it in the terminal'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Directory or file name to search for (e.g. 'jarvis', 'Downloads', 'resume.pdf')"
                    },
                    "open_with": {
                        "type": "string",
                        "description": "App to open the result in: 'vscode', 'dolphin', 'nautilus', 'terminal', 'default', or any app binary name"
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Where to search (default: home directory). E.g. '/home', '/etc', '~'"
                    }
                },
                "required": ["query", "open_with"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "system_volume",
            "description": (
                "Control the system audio volume (not Spotify — the OS-level volume). "
                "Use this for 'increase volume', 'set volume to X%', 'mute', 'unmute', "
                "or 'what is the current volume?'. "
                "This affects all system sounds, not just one app."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "get", "mute", "unmute"],
                        "description": "'set' = set to exact %, 'get' = current level, 'mute'/'unmute'"
                    },
                    "level": {
                        "type": "integer",
                        "description": "Volume percentage 0-100 (required for 'set' action)"
                    }
                },
                "required": ["action"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "save_user_fact",
            "description": (
                "Permanently save a fact you learn about the user. "
                "Call this whenever the user reveals personal information: "
                "their name, location, preferences, job, habits, favourite things, etc. "
                "These facts persist across sessions and are recalled automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short snake_case key (e.g. 'user_name', 'favourite_artist', 'job')"
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact value (e.g. 'Henri', 'Drake', 'software engineer')"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_user_facts",
            "description": (
                "Retrieve all permanently stored facts about the user. "
                "Use this when the user asks what you know about them, "
                "or when you need to recall their personal details."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": (
                "Open any application installed on the system. "
                "Works with browsers, media players, editors, games, system tools — anything. "
                "Tries the binary directly, then Flatpak, then Snap, then xdg-open as fallback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "description": (
                            "Application name or command. "
                            "Examples: 'firefox', 'spotify', 'vlc', 'code', 'discord', "
                            "'gimp', 'steam', 'terminal', 'files'"
                        )
                    },
                    "args": {
                        "type": "string",
                        "description": (
                            "Optional arguments to pass (e.g. a URL or file path). "
                            "For incognito/private mode pass the URL here; "
                            "the incognito flag is set automatically based on the browser."
                        )
                    }
                },
                "required": ["app"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "spotify_control",
            "description": (
                "Control Spotify on the local system. "
                "Can play, pause, skip tracks, adjust volume, search for songs/artists/albums, "
                "and get the currently playing track. "
                "Use 'search' to find and open any song, artist, or album in Spotify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "play_pause", "next", "previous", "stop", "search", "open", "status", "volume"],
                        "description": (
                            "'play' = resume, 'pause' = pause, 'play_pause' = toggle, "
                            "'next' = next track, 'previous' = previous track, "
                            "'stop' = stop, 'search' = search Spotify for query, "
                            "'open' = open Spotify app, 'status' = get current track, "
                            "'volume' = set volume (requires value)"
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for 'search' action (e.g., 'Drake new album', 'Kendrick Lamar')"
                    },
                    "value": {
                        "type": "integer",
                        "description": "Volume level 0-100 (only for 'volume' action)"
                    }
                },
                "required": ["action"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "detect_motion",
            "description": (
                "Detect if motion is currently visible on the webcam. "
                "Simple frame-differencing algorithm. "
                "Returns boolean: true if motion detected, false otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """
    Retrieve a specific tool schema by name.

    Args:
        name: The tool name (e.g., 'web_search')

    Returns:
        The tool schema dict, or empty dict if not found

    Raises:
        Exception: On lookup error (caught internally)
    """
    try:
        for tool in TOOL_SCHEMAS:
            if tool["function"]["name"] == name:
                return tool
        print(f"[Tools] Tool not found: {name}")
        return {}
    except Exception as e:
        print(f"[Tools] Error in get_tool_by_name: {e}")
        return {}


def list_tools() -> List[str]:
    """
    Get a list of all available tool names.

    Returns:
        List of tool names (strings)

    Raises:
        Exception: On iteration error (caught internally)
    """
    try:
        names: List[str] = [tool["function"]["name"] for tool in TOOL_SCHEMAS]
        return names
    except Exception as e:
        print(f"[Tools] Error in list_tools: {e}")
        return []


def tool_schema_as_string() -> str:
    """
    Get a human-readable string of all available tools for display.

    Returns:
        Formatted string with tool names and descriptions

    Raises:
        Exception: On formatting error (caught internally)
    """
    try:
        lines: List[str] = ["=== Available Tools ==="]
        for tool in TOOL_SCHEMAS:
            func = tool["function"]
            name: str = func["name"]
            desc: str = func["description"].split("\n")[0]  # First line only
            lines.append(f"  • {name}: {desc}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Tools] Error in tool_schema_as_string: {e}")
        return ""
