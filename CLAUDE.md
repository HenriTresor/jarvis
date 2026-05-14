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

## Prerequisites

The project requires Ollama running locally (default: `http://localhost:11434`) with these models pulled:

```bash
ollama pull llama3.1:8b   # Main LLM (~4.7 GB)
ollama pull llava:7b      # Vision model, optional (~4.5 GB)
```

## No Test or Lint Setup

There is no pytest, linting, or build configuration in this project. To verify individual modules work:

```bash
python -c "from jarvis.brain.llm_client import LLMClient; LLMClient()"
python -c "from jarvis.agent.agent import JarvisAgent; JarvisAgent()"
python -c "from jarvis.memory.manager import MemoryManager; MemoryManager()"
```

## Architecture

### Interaction Flow

```
User input (voice or text)
  → MemoryManager (retrieve relevant past conversations + facts)
  → JarvisAgent.think() — ReAct loop (max 8 iterations):
      LLM (Ollama/Llama 3.1) → tool_calls? → ToolExecutor → back to LLM
  → final response
  → MemoryManager (save conversation)
  → TextToSpeech (in voice mode)
```

### Module Responsibilities

| Package | Key Class | Role |
|---------|-----------|------|
| `jarvis/brain/` | `LLMClient` | Wraps Ollama; `chat()` and `chat_stream()` |
| `jarvis/agent/` | `JarvisAgent` | ReAct loop; `think(user_input) → str` is the main entry point |
| `jarvis/agent/tools.py` | `TOOLS` | Tool schemas in OpenAI function-calling format |
| `jarvis/agent/tool_executor.py` | `ToolExecutor` | Executes tools by name, returns string results |
| `jarvis/memory/` | `MemoryManager` | ChromaDB (vector search) + SQLite (key-value facts); `build_context()` injects memory into prompts |
| `jarvis/voice/` | `VoicePipeline` | Orchestrates wake word → STT → agent → TTS |
| `jarvis/vision/` | `VisionModule` | OpenCV + LLaVA for image analysis and OCR |
| `jarvis/api/` | FastAPI app | REST + WebSocket; lazy-initializes agent on first request |

### Memory System

Two-layer design in `jarvis/memory/manager.py`:
- **Vector layer** (ChromaDB + `all-MiniLM-L6-v2`): semantic search over conversation history
- **Structured layer** (SQLite): key-value facts (e.g., user name, location)

`build_context(query)` returns a `<memory>...</memory>` XML block injected into LLM prompts.

### Adding a New Tool

1. Add schema dict to `TOOLS` list in `jarvis/agent/tools.py` (OpenAI function-calling format)
2. Add handler in `ToolExecutor.execute_tool()` in `jarvis/agent/tool_executor.py`
3. The agent picks it up automatically — no other changes needed

### API Endpoints

- `POST /chat` — synchronous text chat
- `WS /ws/chat` — streaming chat (token-by-token)
- `GET /memory/facts`, `POST /memory/facts`, `GET /memory/search`
- CORS allows `localhost:5173` and `localhost:3000` (React dev servers)

### LLM Client Portability

`LLMClient` in `jarvis/brain/llm_client.py` uses the Ollama Python SDK but exposes a chat interface compatible with the Anthropic/OpenAI SDK shape. Swapping to the Claude API requires only changes in this file.
