# J.A.R.V.I.S.

A personal AI assistant with a web UI, voice interaction, memory, tool calling, and smart home integration. Powered by Groq cloud inference with multi-provider fallback (Cerebras, Gemini).

---

## Quick Start (Docker)

**Prerequisites:** Docker + Docker Compose installed on Linux.

### 1. Clone and configure

```bash
git clone <repo-url>
cd jarvis
cp .env.example .env
```

Open `.env` and add at minimum your Groq API key:

```env
GROQ_API_KEY=gsk_...        # required — get one free at console.groq.com
GEMINI_API_KEY=...           # optional fallback — aistudio.google.com (1M tokens/day free)
CEREBRAS_API_KEY=...         # optional fallback — cloud.cerebras.ai (free tier)
```

### 2. Start

```bash
docker compose up -d
```

Open **http://localhost:8000** in your browser. That's it.

### 3. Stop

```bash
docker compose down
```

Memory and model caches are stored in Docker named volumes — they persist across restarts and rebuilds.

---

## Voice Mode (Linux)

Voice mode routes audio through PulseAudio/PipeWire on the host.

```bash
# Find your microphone device index
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Add to .env
STT_DEVICE=<index>

# Start with voice overlay
docker compose -f docker-compose.yml -f docker-compose.voice.yml up -d
```

Requires PulseAudio or PipeWire running on the host (standard on any modern Linux desktop).

---

## Configuration

All configuration is via `.env`. See `.env.example` for the full reference.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary LLM model |
| `CEREBRAS_API_KEY` | — | Optional fallback provider |
| `GEMINI_API_KEY` | — | Optional fallback provider |
| `TTS_ENGINE` | `piper` | `piper` (fast, local) or `xtts` (voice cloning) |
| `PIPER_VOICE` | `en_GB-alan-medium` | Piper voice name |
| `SPEAKER_WAV` | — | 6-second WAV for XTTS voice cloning |
| `STT_DEVICE` | `0` | Microphone device index |
| `SPEECH_SPEED` | `1.05` | TTS playback speed multiplier |
| `GMAIL_USER` | — | Gmail address for email tools |
| `GMAIL_APP_PASSWORD` | — | Gmail App Password (not your account password) |
| `SUDO_PASSWORD` | — | Used by system control tools |
| `HOME_ASSISTANT_URL` | — | Home Assistant base URL |
| `HOME_ASSISTANT_TOKEN` | — | Home Assistant long-lived token |
| `LOCATION` | `Kigali, Rwanda` | Injected into system prompt |

---

## Running Without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your keys

python run.py --mode api     # web UI only
python run.py --mode text    # terminal chat
python run.py --mode voice   # full voice pipeline
```

---

## LLM Fallback Chain

When a model is rate-limited, Jarvis automatically tries the next one:

```
Groq llama-3.3-70b  →  Groq qwen3-32b  →  Groq llama-4-scout
→  Cerebras qwen-3-235b  →  Gemini 2.5-flash  →  Gemini 2.0-flash
```

All providers have free tiers. Adding more keys = more headroom before hitting limits.

---

## Tools

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo search |
| `get_weather` | Current weather by city |
| `open_application` | Launch desktop apps |
| `run_code` | Execute Python or shell commands |
| `spotify_control` | Play, pause, skip, volume |
| `system_volume` | Mute, unmute, set volume |
| `smart_home_control` | Home Assistant device control |
| `send_email` | Send Gmail (requires App Password) |
| `get_unread_emails` | Read inbox via IMAP |
| `create_calendar_event` | Add to local calendar |
| `get_upcoming_events` | List upcoming events |
| `set_alarm` | Desktop alarm (fires even if UI is closed) |
| `capture_image` | Webcam snapshot |
| `describe_image` | Vision analysis via LLaVA |
| `get_datetime` | Current time and date |
| `remember_fact` / `recall_fact` | Persistent key-value memory |

---

## Architecture

```
User (voice or web UI)
  → MemoryManager.build_context()     — retrieves past conversations + facts
  → JarvisAgent.think()               — ReAct loop (up to 8 iterations)
      LLMClient.chat()  →  tool_calls?  →  ToolExecutor.execute()
  → final response
  → MemoryManager.save_conversation()
  → TextToSpeech.speak()              — voice mode only
```

```
jarvis/
├── api/           server.py          FastAPI + WebSocket
├── agent/         agent.py           ReAct loop
│                  tools.py           Tool schemas (OpenAI format)
│                  tool_executor.py   Tool implementations
├── brain/         llm_client.py      Multi-provider LLM client
│                  prompts.py         System prompt
├── memory/        manager.py         ChromaDB (vector) + SQLite (facts)
├── voice/         pipeline.py        Voice orchestration
│                  wake_word.py       "Hey Jarvis" detection
│                  stt.py             faster-whisper transcription
│                  tts.py             Piper / XTTS synthesis
├── vision/        vision_module.py   Webcam + LLaVA analysis
└── ui/            index.html         Web frontend
```

---

## Adding a Tool

1. Add schema to `jarvis/agent/tools.py` (OpenAI function-calling format)
2. Add handler method + register in `jarvis/agent/tool_executor.py`
3. Done — the agent picks it up automatically

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/status` | Health check + memory stats |
| `POST` | `/chat` | Synchronous chat (full ReAct loop) |
| `POST` | `/chat/audio` | Chat + base64 WAV audio response |
| `WS` | `/ws/chat` | Streaming chat (token-by-token) |
| `GET` | `/memory/facts` | Read stored facts |
| `POST` | `/memory/facts` | Write a fact |
| `GET` | `/memory/search?q=` | Semantic search over history |
