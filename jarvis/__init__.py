"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System

A local AI assistant with voice, vision, memory, and agent capabilities.
Built entirely on free tools and self-hosted infrastructure.

Modules:
- voice: Wake word detection, speech-to-text, text-to-speech
- brain: LLM client, system prompts, personality
- memory: Vector DB (ChromaDB) + structured facts (SQLite)
- agent: ReAct loop, tool execution, reasoning
- vision: Image analysis, motion detection (LLaVA)
- api: FastAPI backend with REST + WebSocket
"""

__version__ = "1.0.0"
__author__ = "Jarvis Development Team"

print("[Jarvis] Module imported")
