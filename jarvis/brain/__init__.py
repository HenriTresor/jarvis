"""Brain module: LLM client, system prompts, and personality."""

from .llm_client import LLMClient
from .prompts import (
    JARVIS_SYSTEM_PROMPT,
    TOOL_USE_PROMPT,
    ERROR_RECOVERY_PROMPT,
    CONTEXT_BUILDING_PROMPT,
)

__all__ = [
    "LLMClient",
    "JARVIS_SYSTEM_PROMPT",
    "TOOL_USE_PROMPT",
    "ERROR_RECOVERY_PROMPT",
    "CONTEXT_BUILDING_PROMPT",
]
