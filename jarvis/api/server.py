"""
FastAPI Backend Server for J.A.R.V.I.S.

Provides REST API and WebSocket endpoints for chat, memory management,
and real-time streaming responses.

Serves as the bridge between the frontend (React dashboard) and the
Jarvis agent backend (LLM, memory, tools).
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

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


def get_agent() -> JarvisAgent:
    """Get or create the global agent instance."""
    global _agent
    try:
        if _agent is None:
            print(f"[API] Initializing JarvisAgent...")
            _agent = JarvisAgent(location="Kigali, Rwanda")
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

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.

    Returns:
        Dict with API info and status
    """
    try:
        print(f"[API] GET /")
        return {
            "name": "J.A.R.V.I.S. API",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs"
        }
    except Exception as e:
        print(f"[API] Error in root: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            "model": "llama3.1:8b",
            "memory_entries": memory.collection.count(),
            "facts_stored": len(memory.get_all_facts()),
            "conversation_turns": len(agent.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[API] Error in status: {e}")
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

                # Get agent and memory
                agent: JarvisAgent = get_agent()
                memory: MemoryManager = get_memory()

                # Build system prompt
                system: str = agent._build_system_prompt()

                # Build memory context
                memory_ctx: str = memory.build_context(user_msg)
                full_msg: str = (
                    f"{memory_ctx}\n\nUser: {user_msg}"
                    if memory_ctx else user_msg
                )

                # Stream response token by token
                full_response: str = ""
                for chunk in agent.llm.chat_stream(
                    messages=[{"role": "user", "content": full_msg}],
                    system_prompt=system
                ):
                    full_response += chunk
                    await websocket.send_json({
                        "chunk": chunk,
                        "done": False
                    })

                # Signal end of stream
                await websocket.send_json({
                    "chunk": "",
                    "done": True
                })

                # Save conversation to memory
                memory.save_conversation(user_msg, full_response)
                print(f"[API] WebSocket response complete")

            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON",
                    "done": True
                })
            except Exception as e:
                print(f"[API] Error in websocket message loop: {e}")
                await websocket.send_json({
                    "error": str(e),
                    "done": True
                })

    except WebSocketDisconnect:
        print(f"[API] WebSocket disconnected")
    except Exception as e:
        print(f"[API] Error in websocket_chat: {e}")
        try:
            await websocket.send_json({
                "error": str(e),
                "done": True
            })
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
    try:
        print(f"[API] ===== Startup Event =====")
        print(f"[API] J.A.R.V.I.S. API starting...")
        print(f"[API] Available at http://localhost:8000")
        print(f"[API] Docs at http://localhost:8000/docs")
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
