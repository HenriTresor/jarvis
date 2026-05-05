"""
J.A.R.V.I.S. Main Entry Point

Run the Jarvis AI assistant in different modes:
- text: Text-based REPL for testing
- voice: Full voice interaction (wake word → STT → LLM → TTS)
- api: API server only (for frontend integration)

Usage:
    python run.py --mode text    # Text chat
    python run.py --mode voice   # Voice assistant
    python run.py --mode api     # API server
"""

import asyncio
import threading
import uvicorn
import argparse
import time
from typing import Optional

from jarvis.agent.agent import JarvisAgent
from jarvis.voice.pipeline import VoicePipeline
from jarvis.api.server import app


def start_api() -> None:
    """
    Start the FastAPI server in a background thread.

    Runs the API server on localhost:8000 with uvicorn.
    This allows the API to run alongside other modes.

    Returns:
        None
    """
    try:
        print(f"[Run] Starting API server on http://localhost:8000")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="warning"
        )
    except Exception as e:
        print(f"[Run] Error starting API server: {e}")


def text_mode(agent: JarvisAgent) -> None:
    """
    Run Jarvis in text mode (REPL).

    Provides a command-line interface for testing Jarvis without voice.
    Type commands, get responses. Type 'exit' or 'quit' to stop.

    Args:
        agent: The JarvisAgent instance to use for processing

    Returns:
        None
    """
    try:
        print(f"[Run] ===== Text Mode =====")
        print(f"[Run] Type your commands. Type 'exit' or 'quit' to stop.")
        print(f"[Run] API is running in background at http://localhost:8000")
        print(f"[Run] ===================================================")

        while True:
            try:
                user_input: str = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print(f"[Run] Goodbye!")
                    break

                print(f"[Run] Processing: '{user_input[:60]}...'")
                response: str = agent.think(user_input)
                print(f"\nJarvis: {response}")

            except KeyboardInterrupt:
                print(f"\n[Run] Keyboard interrupt. Exiting text mode.")
                break
            except Exception as e:
                print(f"[Run] Error in text mode: {e}")
                continue
    except Exception as e:
        print(f"[Run] Error in text_mode: {e}")


def voice_mode(agent: JarvisAgent) -> None:
    """
    Run Jarvis in voice mode (full voice pipeline).

    Starts the complete voice interaction loop:
    - Listen for wake word ("Hey Jarvis")
    - Record command
    - Process through agent
    - Speak response
    - Repeat

    Args:
        agent: The JarvisAgent instance to use for processing

    Returns:
        None
    """
    try:
        print(f"[Run] ===== Voice Mode =====")
        print(f"[Run] API is running in background at http://localhost:8000")
        print(f"[Run] Say 'Hey Jarvis' to begin...")
        print(f"[Run] Press Ctrl+C to stop.")
        print(f"[Run] ===================================================")

        # Create voice pipeline with agent as brain
        pipeline: VoicePipeline = VoicePipeline(brain_callback=agent.think)

        # Start the pipeline (blocking call)
        pipeline.start()

    except KeyboardInterrupt:
        print(f"\n[Run] Keyboard interrupt. Exiting voice mode.")
    except Exception as e:
        print(f"[Run] Error in voice_mode: {e}")


def api_mode() -> None:
    """
    Run Jarvis in API-only mode.

    Starts the API server and keeps it running.
    Useful for frontend integration or when you only need the API.

    Returns:
        None
    """
    try:
        print(f"[Run] ===== API Mode =====")
        print(f"[Run] API server running at http://localhost:8000")
        print(f"[Run] Docs available at http://localhost:8000/docs")
        print(f"[Run] Press Ctrl+C to stop.")
        print(f"[Run] ===================================================")

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[Run] Keyboard interrupt. Exiting API mode.")

    except Exception as e:
        print(f"[Run] Error in api_mode: {e}")


def main() -> None:
    """
    Main entry point for J.A.R.V.I.S.

    Parses command line arguments and starts the appropriate mode.
    Always starts the API server in a background thread.

    Returns:
        None
    """
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="J.A.R.V.I.S. - Local AI Assistant"
        )
        parser.add_argument(
            "--mode",
            choices=["text", "voice", "api"],
            default="text",
            help="Run mode: text (REPL), voice (full assistant), api (server only)"
        )
        parser.add_argument(
            "--location",
            default="Kigali, Rwanda",
            help="User location for context (default: Kigali, Rwanda)"
        )
        parser.add_argument(
            "--ha-url",
            help="Home Assistant URL (e.g., http://localhost:8123)"
        )
        parser.add_argument(
            "--ha-token",
            help="Home Assistant long-lived token"
        )

        args: argparse.Namespace = parser.parse_args()

        print(f"[Run] ===== J.A.R.V.I.S. v1.0.0 =====")
        print(f"[Run] Mode: {args.mode}")
        print(f"[Run] Location: {args.location}")

        # Start API server in background thread
        api_thread: threading.Thread = threading.Thread(
            target=start_api,
            daemon=True
        )
        api_thread.start()

        # Give API a moment to start
        time.sleep(2)

        # Initialize agent (only for text/voice modes)
        agent: Optional[JarvisAgent] = None
        if args.mode in ("text", "voice"):
            try:
                print(f"[Run] Initializing JarvisAgent...")
                agent = JarvisAgent(
                    home_assistant_url=args.ha_url,
                    ha_token=args.ha_token,
                    location=args.location
                )
                print(f"[Run] Agent ready.")
            except Exception as e:
                print(f"[Run] Failed to initialize agent: {e}")
                print(f"[Run] Make sure Ollama is running and models are pulled.")
                return

        # Run the selected mode
        if args.mode == "text":
            text_mode(agent)
        elif args.mode == "voice":
            voice_mode(agent)
        elif args.mode == "api":
            api_mode()

        # Cleanup
        if agent:
            agent.close()

        print(f"[Run] J.A.R.V.I.S. shutdown complete.")

    except KeyboardInterrupt:
        print(f"\n[Run] Keyboard interrupt. Shutting down...")
    except Exception as e:
        print(f"[Run] Error in main: {e}")


if __name__ == "__main__":
    main()
