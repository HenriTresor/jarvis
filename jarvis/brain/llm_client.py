"""
LLM Client for J.A.R.V.I.S. Brain

Wraps Groq for fast cloud inference using Llama 3.3 70B (or any Groq model).
Supports both single-call and streaming interactions.
"""

import os
import re
from groq import Groq
from typing import List, Dict, Any, Optional, Generator
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Wraps Groq cloud inference with an automatic fallback model chain.

    Tries each model in FALLBACK_CHAIN in order when rate limits are hit.
    When all models are exhausted, returns a Jarvis-style message with the
    actual retry time parsed from the Groq error response.
    """

    MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ordered list of fallback models to try when the primary is rate-limited.
    # Each has a different free-tier quota so they fail independently.
    FALLBACK_CHAIN: List[str] = [
        m.strip()
        for m in os.getenv(
            "GROQ_FALLBACK_CHAIN",
            """
            llama-3.3-70b-versatile,
            openai/gpt-oss-20b,
            openai/gpt-oss-120b,
            groq/compound,
            meta-llama/llama-prompt-guard-2-22m,
            meta-llama/llama-prompt-guard-2-86m,
            meta-llama/llama-4-scout-17b-16e-instruct,
            llama-3.1-8b-instant,
            qwen/qwen3-32b,
            """,
        ).split(",")
        if m.strip()
    ]


    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "[LLM] GROQ_API_KEY not set. Add it to your .env file."
            )
        self.client: Groq = Groq(api_key=api_key)
        print(f"[LLM] Connected to Groq. Model: {self.MODEL}")

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip non-standard keys before sending to the Groq API."""
        cleaned = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                cleaned.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", "0"),
                })
            elif role == "assistant" and msg.get("tool_calls"):
                cleaned.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": msg["tool_calls"],
                })
            else:
                cleaned.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })
        return cleaned

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Single, non-streaming call to the LLM.

        Returns a dict with:
          - "content": assistant text response
          - "tool_calls": list of tool call dicts (if any)
          - "model": model name
        """
        kwargs: Dict[str, Any] = {"model": self.MODEL, "messages": []}
        try:
            formatted: List[Dict[str, Any]] = []
            if system_prompt:
                formatted.append({"role": "system", "content": system_prompt})
            formatted.extend(self._clean_messages(messages))

            kwargs = {"model": self.MODEL, "messages": formatted}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
                print(f"[LLM] Tools available: {len(tools)}")

            print(f"[LLM] Sending request (messages: {len(messages)})")
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            result: Dict[str, Any] = {
                "content": msg.content or "",
                "model": self.MODEL,
            }

            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                print(f"[LLM] {len(msg.tool_calls)} tool call(s) returned.")

            print(f"[LLM] Response received.")
            return result

        except Exception as e:
            err_str = str(e)
            last_error = err_str

            if "rate_limit_exceeded" in err_str:
                tried = {kwargs.get("model", self.MODEL)}
                for fallback in self.FALLBACK_CHAIN:
                    if fallback in tried:
                        continue
                    tried.add(fallback)
                    print(f"[LLM] Rate limit hit. Trying fallback: {fallback}...")
                    try:
                        kwargs["model"] = fallback
                        # If previous error was "request too large", trim history
                        # to the 6 most recent messages so smaller models can handle it
                        if "too large" in last_error or "reduce your message size" in last_error:
                            msgs = kwargs["messages"]
                            # Always keep the system message (first) + last 6 messages
                            system_msgs = [m for m in msgs if m.get("role") == "system"]
                            non_system = [m for m in msgs if m.get("role") != "system"]
                            kwargs["messages"] = system_msgs + non_system[-6:]
                            print(f"[LLM] Trimmed history to last 6 messages for {fallback}.")
                        response = self.client.chat.completions.create(**kwargs)
                        msg = response.choices[0].message
                        result: Dict[str, Any] = {
                            "content": msg.content or "",
                            "model": fallback,
                        }
                        if msg.tool_calls:
                            result["tool_calls"] = [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ]
                        print(f"[LLM] Fallback {fallback} responded.")
                        return result
                    except Exception as fe:
                        last_error = str(fe)
                        print(f"[LLM] Fallback {fallback} failed: {last_error[:120]}")
                        # Only stop the chain on auth/credential failures.
                        # Everything else (model unavailable, unsupported features,
                        # too large, rate limits) — keep trying the next model.
                        unrecoverable = any(t in last_error.lower() for t in (
                            "invalid api key", "invalid_api_key",
                            "authentication", "unauthorized", "forbidden",
                        ))
                        if unrecoverable:
                            break

                # All models exhausted — extract retry time and respond in character
                retry_msg = self._parse_retry_time(last_error)
                print(f"[LLM] All models rate-limited. {retry_msg}")
                return {
                    "content": (
                        f"All systems are currently throttled, sir. "
                        f"{retry_msg} I'll be fully operational again shortly."
                    ),
                    "model": "none",
                }

            print(f"[LLM] Error in chat: {e}")
            return {
                "content": "Systems encountered an unexpected error, sir. Standing by.",
                "model": self.MODEL,
            }

    def _parse_retry_time(self, error_str: str) -> str:
        """Extract the retry-after time from a Groq rate limit error string."""
        match = re.search(r"Please try again in ([^\.']+)", error_str)
        if match:
            return f"Groq requests a {match.group(1).strip()} cooldown."
        return "Retry time unknown."

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> Generator[str, None, None]:
        """
        Streaming call to the LLM. Yields text chunks as they arrive.

        Note: streaming does not support tool calls — use chat() for tool use.
        """
        formatted: List[Dict[str, Any]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend(self._clean_messages(messages))

        models_to_try = [self.MODEL] + self.FALLBACK_CHAIN
        last_error = ""

        for model in models_to_try:
            try:
                print(f"[LLM] Streaming request via {model} (messages: {len(messages)})")
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=formatted,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                print(f"[LLM] Streaming complete.")
                return
            except Exception as e:
                last_error = str(e)
                unrecoverable = any(t in last_error.lower() for t in (
                    "invalid api key", "invalid_api_key",
                    "authentication", "unauthorized", "forbidden",
                ))
                if unrecoverable:
                    print(f"[LLM] Auth error on {model} — stopping chain.")
                    yield f"Authentication error, sir. Please check the API key."
                    return
                print(f"[LLM] {model} failed: {last_error[:120]} — trying next...")
                continue

        retry_msg = self._parse_retry_time(last_error)
        print(f"[LLM] All streaming models rate-limited. {retry_msg}")
        yield (
            f"All systems are currently throttled, sir. "
            f"{retry_msg} I'll be back online shortly."
        )
