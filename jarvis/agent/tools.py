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
                "Search the web for current information, news, or facts. "
                "Use this to find up-to-date information not in your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'latest Python news')"
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
                "Execute Python code and return the output. "
                "Use for calculations, data processing, or running scripts. "
                "WARNING: Only runs trusted code. No network access."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute (e.g., 'print(2 + 2)')"
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
                "Read the contents of a text file from the filesystem. "
                "Files must be in ~/jarvis_files or subdirectories. "
                "Returns first 2000 characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (e.g., 'documents/notes.txt')"
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
