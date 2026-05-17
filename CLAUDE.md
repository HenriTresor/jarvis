# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Text mode — no audio hardware needed, best for development/testing
python run.py --mode text

# Voice mode — full pipeline with wake word
python run.py --mode voice --location "City, Country"

# API server only
python run.py --mode api

# With Home Assistant smart home integration
python run.py --mode voice --ha-url http://localhost:8123 --ha-token YOUR_TOKEN
```

All modes always start the FastAPI server in a background thread on port 8000. The UI is served at `GET /` from `ui/index.html`. API docs at `http://localhost:8000/docs`.

## Prerequisites

Ollama must be running locally (default: `http://localhost:11434`):

```bash
ollama pull llama3.2:3b   # Main LLM — default model (~2 GB)
ollama pull llava:7b      # Vision model — optional (~4.5 GB)
```

The active model is controlled by `OLLAMA_MODEL` env var (default `llama3.2:3b`, set in `LLMClient.MODEL`).

## Environment Variables

All are optional — CLI args take precedence over env vars:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model to use |
| `JARVIS_MEMORY_PATH` | `./jarvis_memory` | ChromaDB + SQLite storage directory |
| `LOCATION` | `Kigali, Rwanda` | Injected into system prompt |
| `HOME_ASSISTANT_URL` | — | Home Assistant base URL |
| `HOME_ASSISTANT_TOKEN` | — | Home Assistant long-lived token |
| `SPEAKER_WAV` | — | Path to 6-second WAV for TTS voice cloning |
| `CAMERA_INDEX` | `0` | OpenCV camera index |
| `JARVIS_HOST` / `JARVIS_PORT` | `0.0.0.0` / `8000` | API server bind address |

## No Test or Lint Setup

There is no pytest, linting, or build configuration. To verify individual modules:

```bash
python -c "from jarvis.brain.llm_client import LLMClient; LLMClient()"
python -c "from jarvis.agent.agent import JarvisAgent; JarvisAgent()"
python -c "from jarvis.memory.manager import MemoryManager; MemoryManager()"
```

## Architecture

### Interaction Flow

```
User input (voice or text)
  → MemoryManager.build_context() — retrieves relevant past conversations + facts
  → JarvisAgent.think() — ReAct loop (max 8 iterations):
      LLM (Ollama) → tool_calls? → ToolExecutor.execute() → back to LLM
  → final response text
  → MemoryManager.save_conversation()
  → TextToSpeech.speak() (voice mode only)
```

### Module Responsibilities

| Package | Key Class | Role |
|---|---|---|
| `jarvis/brain/llm_client.py` | `LLMClient` | Wraps Ollama; `chat()` (blocking) and `chat_stream()` (generator) |
| `jarvis/brain/prompts.py` | `JARVIS_SYSTEM_PROMPT` | System prompt template with `{datetime}` and `{location}` placeholders |
| `jarvis/agent/agent.py` | `JarvisAgent` | ReAct loop; `think(user_input) → str` is the main entry point |
| `jarvis/agent/tools.py` | `TOOL_SCHEMAS` | Tool schemas in OpenAI function-calling format |
| `jarvis/agent/tool_executor.py` | `ToolExecutor` | Routes tool names to handler methods; returns strings |
| `jarvis/memory/manager.py` | `MemoryManager` | ChromaDB (vector) + SQLite (facts); `build_context()` returns `<memory>` XML block |
| `jarvis/voice/pipeline.py` | `VoicePipeline` | Orchestrates wake word → STT → agent → TTS |
| `jarvis/voice/wake_word.py` | `WakeWordDetector` | OpenWakeWord, listens for "hey_jarvis" on a background thread |
| `jarvis/voice/stt.py` | `SpeechToText` | faster-whisper (`base` model, CPU int8); records until silence then transcribes |
| `jarvis/voice/tts.py` | `TextToSpeech` | Coqui XTTS v2 (~1.8 GB, downloads on first run); supports voice cloning via `SPEAKER_WAV` |
| `jarvis/vision/vision_module.py` | `VisionModule` | OpenCV webcam capture + LLaVA analysis via Ollama; lazy-loaded at startup |
| `jarvis/api/server.py` | FastAPI `app` | REST + WebSocket; agent and memory are lazy-initialized on first request |

### Memory System

Two-layer design in `jarvis/memory/manager.py`:
- **Vector layer** (ChromaDB + `all-MiniLM-L6-v2`): semantic search over conversation history
- **Structured layer** (SQLite `facts.db`): key-value facts (e.g., user name, location)

`build_context(query)` combines both layers into a `<memory>...</memory>` XML block prepended to each user message.

### Adding a New Tool

1. Add schema dict to `TOOL_SCHEMAS` list in `jarvis/agent/tools.py` (OpenAI function-calling format)
2. Add a `_handler_name` method and register it in the `handlers` dict inside `ToolExecutor.execute()` in `jarvis/agent/tool_executor.py`
3. The agent picks it up automatically — no other changes needed

### API Endpoints

- `GET /` — serves `ui/index.html`
- `GET /status` — health check with memory stats
- `POST /chat` — synchronous text chat (full ReAct loop)
- `POST /chat/audio` — same as `/chat` but also returns base64 WAV audio for browser playback
- `WS /ws/chat` — streaming chat token-by-token (bypasses ReAct tool loop; uses `chat_stream` directly)
- `GET /memory/facts`, `POST /memory/facts` — read/write structured facts
- `GET /memory/search?q=...` — semantic search over conversation history
- CORS allows `localhost:5173` and `localhost:3000`

### Unimplemented Tools

`send_email`, `get_unread_emails`, `get_upcoming_events`, and `create_calendar_event` currently return `[SIMULATED]` responses. Google API OAuth integration is stubbed but not implemented. The `smart_home_control` tool falls back to simulated responses when `HOME_ASSISTANT_URL` is not configured.

### LLM Client Portability

`LLMClient` exposes a `chat()` interface compatible with Anthropic/OpenAI SDK shapes. Swapping to the Claude API requires only changes in `jarvis/brain/llm_client.py` — the agent, tools, and memory layers are provider-agnostic.
