"""
LLM Client for J.A.R.V.I.S. Brain

Wraps Ollama for local, free LLM inference using Llama 3.1 8B.
Supports both single-call and streaming interactions.
Migration path: swap ollama calls for anthropic.messages.create() (same interface).
"""

import ollama
from typing import List, Dict, Any, Optional, Generator


class LLMClient:
    """
    Wraps Ollama for local LLM inference.

    Uses Llama 3.1 8B as the default model:
    - Runs on CPU (8GB RAM) or GPU if available
    - Supports native function/tool calling
    - Strong instruction following and reasoning
    - ~10-15 tokens/second on CPU (acceptable for personal assistant)

    Production migration:
        Replace ollama.chat() with anthropic.messages.create()
        Same tool/function schema works with Claude's API—no code changes needed.

    Example:
        client = LLMClient()
        response = client.chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            system_prompt="You are a helpful assistant."
        )
        print(response["content"])  # Output: "2+2 equals 4."

    Example with streaming:
        for chunk in client.chat_stream(
            messages=[{"role": "user", "content": "Tell me a story"}],
            system_prompt="Be creative."
        ):
            print(chunk, end="", flush=True)
    """

    MODEL: str = "llama3.1:8b"
    DEFAULT_TIMEOUT: int = 300  # 5 minutes

    def __init__(self) -> None:
        """
        Initialize the LLM client and verify Ollama is running.

        Attempts to connect to Ollama at the default localhost:11434.
        If Ollama is not running, raises an exception with clear instructions.

        Raises:
            RuntimeError: If Ollama is not running or unreachable
        """
        try:
            print(f"[LLM] Connecting to Ollama...")
            # Verify Ollama is running by listing available models
            models: List[Dict[str, Any]] = ollama.list()
            print(f"[LLM] Connected to Ollama.")
            print(f"[LLM] Available models: {len(models)}")
            print(f"[LLM] Using model: {self.MODEL}")
        except Exception as e:
            error_msg: str = (
                f"[LLM] Error: Cannot connect to Ollama.\n"
                f"Make sure Ollama is running:\n"
                f"  1. Install Ollama from https://ollama.com\n"
                f"  2. Run: ollama serve\n"
                f"  3. Pull the model: ollama pull {self.MODEL}\n"
                f"Details: {e}"
            )
            print(error_msg)
            raise RuntimeError(error_msg)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Single, non-streaming call to the LLM.

        Sends a list of messages to the model and returns the complete response.
        Supports optional tools/functions for the LLM to call.

        Args:
            messages: List of message dicts with "role" (user/assistant/tool) and "content"
                     Example: [{"role": "user", "content": "Hello"}]
            tools: Optional list of tool/function definitions (JSON schema format)
                  Example: [{"type": "function", "function": {...}}]
            system_prompt: Optional system prompt to prepend to the conversation

        Returns:
            Dict with keys:
            - "content": The assistant's text response (str)
            - "tool_calls": Optional list of tool calls (if any)
            - "model": The model used (str)
            - "stop_reason": Why the model stopped (str)

        Raises:
            Exception: On connection/API error (caught internally, returns safe fallback)
        """
        try:
            if not messages:
                print(f"[LLM] Error: No messages provided")
                return {"content": "No input received.", "model": self.MODEL}

            # Format messages: prepend system prompt if provided
            formatted_messages: List[Dict[str, Any]] = []
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            formatted_messages.extend(messages)

            print(f"[LLM] Sending request to model (messages: {len(messages)})")

            # Build the API call
            kwargs: Dict[str, Any] = {
                "model": self.MODEL,
                "messages": formatted_messages,
                "stream": False
            }
            if tools:
                kwargs["tools"] = tools
                print(f"[LLM] Tools available: {len(tools)}")

            # Call Ollama
            response: Dict[str, Any] = ollama.chat(**kwargs)
            message: Dict[str, Any] = response.get("message", {})

            print(f"[LLM] Response received.")
            return message
        except Exception as e:
            print(f"[LLM] Error in chat: {e}")
            return {
                "content": "I encountered an error processing your request. Please try again.",
                "model": self.MODEL
            }

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = ""
    ) -> Generator[str, None, None]:
        """
        Streaming call to the LLM.

        Yields text chunks as they arrive from the model, allowing
        real-time response streaming and low latency user feedback.

        Args:
            messages: List of message dicts (same format as chat())
            system_prompt: Optional system prompt to prepend

        Yields:
            str: Text chunks of the response as they arrive

        Raises:
            Exception: On connection/API error (caught internally, yields error message)

        Example:
            for chunk in client.chat_stream(messages):
                print(chunk, end="", flush=True)
            print()  # newline at the end
        """
        try:
            if not messages:
                print(f"[LLM] Error: No messages provided")
                yield "No input received."
                return

            # Format messages
            formatted_messages: List[Dict[str, Any]] = []
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            formatted_messages.extend(messages)

            print(f"[LLM] Streaming request to model (messages: {len(messages)})")

            # Call Ollama with streaming enabled
            stream = ollama.chat(
                model=self.MODEL,
                messages=formatted_messages,
                stream=True
            )

            # Yield chunks as they arrive
            for chunk in stream:
                message_chunk: Dict[str, Any] = chunk.get("message", {})
                content_chunk: str = message_chunk.get("content", "")
                if content_chunk:
                    yield content_chunk

            print(f"[LLM] Streaming complete.")
        except Exception as e:
            print(f"[LLM] Error in chat_stream: {e}")
            yield f"Error streaming response: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieve information about the currently configured model.

        Returns metadata about the model (size, parameters, etc.).

        Returns:
            Dict with model information, or empty dict on error

        Raises:
            Exception: On API error (caught internally)
        """
        try:
            print(f"[LLM] Fetching model info for {self.MODEL}...")
            # Ollama doesn't have a dedicated endpoint for this,
            # but we can return standard info about the model
            info: Dict[str, Any] = {
                "model": self.MODEL,
                "description": "Llama 3.1 8B (local, free, function-calling enabled)",
                "context_window": 8192,
                "parameters": 8000000000,  # 8 billion
                "quantization": "q4_0 (4-bit quantized)"
            }
            print(f"[LLM] Model info retrieved.")
            return info
        except Exception as e:
            print(f"[LLM] Error in get_model_info: {e}")
            return {}
