"""
FastAPI Backend Server for J.A.R.V.I.S.

Provides REST API and WebSocket endpoints for chat, memory management,
and real-time streaming responses.

Serves as the bridge between the frontend (React dashboard) and the
Jarvis agent backend (LLM, memory, tools).
"""

import asyncio
import os
import json
import base64
import threading
import queue
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agent.agent import JarvisAgent
from ..memory.manager import MemoryManager


# ─────────────────────────────────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────────────────────────────────

app: FastAPI = FastAPI(
    title="J.A.R.V.I.S. API",
    description="Local AI Assistant API",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[API] FastAPI app initialized")

# ─────────────────────────────────────────────────────────────────────
# Global Instances (lazy-initialized on first request)
# ─────────────────────────────────────────────────────────────────────

_agent: Optional[JarvisAgent] = None
_memory: Optional[MemoryManager] = None
_tts = None  # Lazy-initialized on first /chat/audio request


def get_agent() -> JarvisAgent:
    """Get or create the global agent instance."""
    global _agent
    try:
        if _agent is None:
            print(f"[API] Initializing JarvisAgent...")
            _agent = JarvisAgent(
                location=os.getenv("LOCATION", "Kigali, Rwanda"),
                home_assistant_url=os.getenv("HOME_ASSISTANT_URL") or None,
                ha_token=os.getenv("HOME_ASSISTANT_TOKEN") or None,
            )
        return _agent
    except Exception as e:
        print(f"[API] Error in get_agent: {e}")
        raise


def get_memory() -> MemoryManager:
    """Get or create the global memory instance."""
    global _memory
    try:
        if _memory is None:
            print(f"[API] Initializing MemoryManager...")
            _memory = MemoryManager()
        return _memory
    except Exception as e:
        print(f"[API] Error in get_memory: {e}")
        raise


def get_tts():
    """Get or create the global TTS instance (lazy, heavy init)."""
    global _tts
    try:
        if _tts is None:
            from ..voice.tts import TextToSpeech
            print(f"[API] Initializing TextToSpeech...")
            _tts = TextToSpeech()
        return _tts
    except Exception as e:
        print(f"[API] TTS init failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "What's the weather?"}
        }


class FactRequest(BaseModel):
    """Request model for fact storage."""
    key: str
    value: str

    class Config:
        json_schema_extra = {
            "example": {
                "key": "user_name",
                "value": "Alice"
            }
        }


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    query: str
    max_results: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "query": "weather",
                "max_results": 3
            }
        }


# ─────────────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Jarvis UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "index.html")
    ui_path = os.path.normpath(ui_path)
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>J.A.R.V.I.S. API</h1><p>UI not found. See <a href='/docs'>/docs</a>.</p>")


@app.get("/status")
async def status() -> Dict[str, Any]:
    """
    System health check and status.

    Returns:
        Dict with system status, model info, memory stats
    """
    try:
        print(f"[API] GET /status")
        memory: MemoryManager = get_memory()
        agent: JarvisAgent = get_agent()

        return {
            "status": "online",
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "location": os.getenv("LOCATION", "Kigali, Rwanda"),
            "memory_entries": memory.collection.count(),
            "facts_stored": len(memory.get_all_facts()),
            "conversation_turns": len(agent.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[API] Error in status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """System hardware metrics — CPU, RAM, disk, network."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()

        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for k in ('coretemp', 'k10temp', 'cpu_thermal', 'acpitz'):
                    if k in temps and temps[k]:
                        cpu_temp = round(temps[k][0].current, 1)
                        break
        except Exception:
            pass

        return {
            "cpu_percent": round(cpu, 1),
            "cpu_count": psutil.cpu_count(logical=True),
            "mem_percent": round(mem.percent, 1),
            "mem_used_gb": round(mem.used / 1024**3, 2),
            "mem_total_gb": round(mem.total / 1024**3, 2),
            "swap_percent": round(swap.percent, 1),
            "disk_percent": round(disk.percent, 1),
            "disk_used_gb": round(disk.used / 1024**3, 1),
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "net_sent_bytes": net.bytes_sent,
            "net_recv_bytes": net.bytes_recv,
            "cpu_temp": cpu_temp,
            "uptime": int(time.time() - psutil.boot_time()),
            "process_count": len(psutil.pids()),
        }
    except Exception as e:
        print(f"[API] Error in get_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Text-based chat with Jarvis.

    Sends the user message to the agent, which processes it through
    the ReAct loop (reasoning, tool calling, etc.) and returns a response.

    Args:
        request: ChatRequest with message field

    Returns:
        Dict with response text and metadata

    Raises:
        HTTPException: On processing error
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )

        print(f"[API] POST /chat: {request.message[:60]}...")

        agent: JarvisAgent = get_agent()
        response: str = agent.think(request.message)

        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(request: ChatRequest) -> Dict[str, Any]:
    """
    Chat with Jarvis and receive both text response and audio as base64 WAV.

    The browser uses the audio bytes to play Jarvis's voice and visualize the waveform.

    Returns:
        Dict with response text, base64-encoded WAV audio, and metadata
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        print(f"[API] POST /chat/audio: {request.message[:60]}...")

        agent: JarvisAgent = get_agent()
        response_text: str = agent.think(request.message)

        # Generate TTS audio
        audio_b64: Optional[str] = None
        tts = get_tts()
        if tts:
            try:
                audio_bytes: bytes = tts.generate_audio_bytes(response_text)
                if audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            except Exception as tts_err:
                print(f"[API] TTS error (non-fatal): {tts_err}")

        return {
            "response": response_text,
            "audio": audio_b64,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error in chat_audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/facts")
async def get_facts() -> Dict[str, Any]:
    """
    Get all stored facts about the user.

    Returns:
        Dict with fact keys and values
    """
    try:
        print(f"[API] GET /memory/facts")
        memory: MemoryManager = get_memory()
        facts: Dict[str, str] = memory.get_all_facts()

        return {
            "facts": facts,
            "count": len(facts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[API] Error in get_facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/facts")
async def set_fact(request: FactRequest) -> Dict[str, Any]:
    """
    Store or update a fact about the user.

    Examples:
    - {"key": "user_name", "value": "Alice"}
    - {"key": "user_location", "value": "Kigali"}
    - {"key": "preferred_language", "value": "English"}

    Args:
        request: FactRequest with key and value

    Returns:
        Dict with status and stored fact
    """
    try:
        if not request.key or not request.value:
            raise HTTPException(
                status_code=400,
                detail="Key and value cannot be empty"
            )

        print(f"[API] POST /memory/facts: {request.key}")

        memory: MemoryManager = get_memory()
        memory.set_fact(request.key, request.value)

        return {
            "status": "saved",
            "key": request.key,
            "value": request.value,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error in set_fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search")
async def search_memory(q: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search past conversations by semantic similarity.

    Finds conversations that are semantically similar to the query.
    Useful for "remind me what I asked about weather" type queries.

    Query parameters:
    - q: Search query (required)
    - max_results: Number of results to return (default: 5, max: 10)

    Args:
        q: Search query string
        max_results: Maximum results to return

    Returns:
        Dict with search results
    """
    try:
        if not q or not q.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        max_results = min(max_results, 10)  # Cap at 10
        print(f"[API] GET /memory/search: {q}")

        memory: MemoryManager = get_memory()
        results: List[str] = memory.retrieve_relevant(q, n_results=max_results)

        return {
            "query": q,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error in search_memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────
# WebSocket Endpoints (Real-time Streaming)
# ─────────────────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time streaming chat.

    Client sends: {"message": "Your question here"}
    Server sends chunks: {"chunk": "text", "done": false}
    Final message: {"chunk": "", "done": true}

    Allows real-time response streaming and low-latency feedback.

    Args:
        websocket: WebSocket connection from client

    Returns:
        None (runs until connection closes)
    """
    try:
        await websocket.accept()
        print(f"[API] WebSocket connection accepted")

        while True:
            try:
                # Receive message from client
                data: Dict[str, Any] = await websocket.receive_json()
                user_msg: str = data.get("message", "").strip()

                if not user_msg:
                    await websocket.send_json({
                        "error": "Empty message",
                        "done": True
                    })
                    continue

                print(f"[API] WebSocket message: {user_msg[:60]}...")

                agent: JarvisAgent = get_agent()

                # Bridge the sync think_stream() generator to the async WebSocket
                # using a thread + queue so tools and streaming both work.
                chunk_queue: queue.Queue = queue.Queue()

                def _run_stream() -> None:
                    try:
                        for chunk in agent.think_stream(user_msg):
                            chunk_queue.put(chunk)
                    except Exception as exc:
                        chunk_queue.put(exc)
                    finally:
                        chunk_queue.put(None)  # sentinel

                stream_thread = threading.Thread(target=_run_stream, daemon=True)
                stream_thread.start()

                loop = asyncio.get_event_loop()
                full_text: str = ""

                while True:
                    item = await loop.run_in_executor(None, chunk_queue.get)
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        await websocket.send_json({"error": str(item), "done": True})
                        break
                    if isinstance(item, dict):
                        # Structured event (e.g. tool_activity) — forward as-is, not text
                        await websocket.send_json(item)
                        continue
                    full_text += item
                    await websocket.send_json({"chunk": item, "done": False})

                audio_b64: Optional[str] = None
                tts = get_tts()
                if tts and full_text.strip():
                    try:
                        audio_bytes: bytes = await loop.run_in_executor(
                            None, tts.generate_audio_bytes, full_text.strip()
                        )
                        if audio_bytes:
                            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    except Exception as tts_err:
                        print(f"[API] WebSocket TTS error (non-fatal): {tts_err}")

                await websocket.send_json({"chunk": "", "done": True, "audio": audio_b64})
                print(f"[API] WebSocket response complete")

            except WebSocketDisconnect:
                return
            except json.JSONDecodeError:
                try:
                    await websocket.send_json({"error": "Invalid JSON", "done": True})
                except Exception:
                    pass
            except RuntimeError as e:
                es = str(e).lower()
                if any(k in es for k in ("shutdown", "closed", "disconnect", "receive")):
                    print(f"[API] WebSocket connection closed")
                    return
                print(f"[API] Error in websocket message loop: {e}")
                try:
                    await websocket.send_json({"error": str(e), "done": True})
                except Exception:
                    pass
            except Exception as e:
                es = str(e).lower()
                if any(k in es for k in ("disconnect", "connection closed", "receive")):
                    return
                print(f"[API] Error in websocket message loop: {e}")
                try:
                    await websocket.send_json({"error": str(e), "done": True})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[API] WebSocket disconnected")
    except RuntimeError as e:
        if "shutdown" not in str(e).lower() and "closed" not in str(e).lower():
            print(f"[API] Error in websocket_chat: {e}")
    except Exception as e:
        print(f"[API] Error in websocket_chat: {e}")
        try:
            await websocket.send_json({"error": str(e), "done": True})
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions with consistent format.

    Args:
        request: The request that caused the error
        exc: The HTTPException

    Returns:
        JSON response with error details
    """
    try:
        print(f"[API] HTTP Exception {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        print(f"[API] Error in exception handler: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat()
            }
        )


# ─────────────────────────────────────────────────────────────────────
# Startup/Shutdown Events
# ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """
    Handle startup event.

    Initializes the agent and memory on first request.

    Returns:
        None
    """
    global _memory, _agent
    try:
        print(f"[API] ===== Startup Event =====")
        print(f"[API] J.A.R.V.I.S. API starting...")
        print(f"[API] Available at http://localhost:8000")
        print(f"[API] Docs at http://localhost:8000/docs")
        # Initialize in the main thread so SQLite connections are thread-safe.
        _memory = MemoryManager()
        print(f"[API] MemoryManager ready.")
        _agent = JarvisAgent(
            location=os.getenv("LOCATION", "Kigali, Rwanda"),
            home_assistant_url=os.getenv("HOME_ASSISTANT_URL") or None,
            ha_token=os.getenv("HOME_ASSISTANT_TOKEN") or None,
        )
        print(f"[API] JarvisAgent ready.")
    except Exception as e:
        print(f"[API] Error in startup_event: {e}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Handle shutdown event.

    Cleans up resources: closes database connections, etc.

    Returns:
        None
    """
    try:
        print(f"[API] ===== Shutdown Event =====")
        global _agent, _memory

        if _memory:
            _memory.close()
            print(f"[API] Memory closed")

        if _agent:
            _agent.close()
            print(f"[API] Agent closed")

        print(f"[API] J.A.R.V.I.S. API shutting down...")
    except Exception as e:
        print(f"[API] Error in shutdown_event: {e}")


if __name__ == "__main__":
    import uvicorn
    print(f"[API] Starting server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
