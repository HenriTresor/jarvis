"""
LLM Client for J.A.R.V.I.S. Brain

Primary provider: Groq (Llama 3.3 70B and fallbacks).
Secondary providers: Cerebras and Google Gemini (OpenAI-compatible endpoints).
Falls through all providers before giving up.
"""

import os
import re
from groq import Groq
from typing import List, Dict, Any, Optional, Generator
from dotenv import load_dotenv

load_dotenv()


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> reasoning blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_tool_calls(msg) -> Optional[List[Dict[str, Any]]]:
    if not msg.tool_calls:
        return None
    return [
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


class LLMClient:
    """
    Multi-provider LLM client with automatic fallback.

    Order: Groq models → Cerebras → Gemini.
    Each provider is skipped silently if its API key is absent.
    """

    MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # 8B and smaller models are unreliable with tool calling — they hallucinate
    # tool invocations from conversation context. Strip tools for these models.
    NO_TOOLS_MODELS: set = {
        "llama-3.1-8b-instant",
        "llama3.1-8b",
    }

    # Only tool/function-calling capable models confirmed on Groq's free tier.
    # Ordered strongest → weakest. All confirmed tool-callable on Groq free tier.
    GROQ_CHAIN: List[str] = [
        m.strip()
        for m in os.getenv(
            "GROQ_FALLBACK_CHAIN",
            """
            llama-3.3-70b-versatile,
            qwen/qwen3-32b,
            meta-llama/llama-4-scout-17b-16e-instruct,
            llama-3.1-8b-instant,
            """,
        ).split(",")
        if m.strip()
    ]

    # Kept for backwards compatibility — callers that read FALLBACK_CHAIN still work.
    FALLBACK_CHAIN = GROQ_CHAIN

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("[LLM] GROQ_API_KEY not set. Add it to your .env file.")
        self.groq = Groq(api_key=api_key)

        # Optional secondary providers — imported lazily so missing package doesn't crash.
        self._alt_providers: List[Dict[str, Any]] = []
        self._init_alt_providers()

        providers = ["Groq"] + [p["name"] for p in self._alt_providers]
        print(f"[LLM] Providers: {', '.join(providers)}. Primary model: {self.MODEL}")

    def _init_alt_providers(self) -> None:
        try:
            from openai import OpenAI as _OAI
        except ImportError:
            print("[LLM] openai package not installed — Cerebras/Gemini unavailable.")
            return

        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        if cerebras_key:
            self._alt_providers.append({
                "name": "Cerebras",
                "client": _OAI(api_key=cerebras_key, base_url="https://api.cerebras.ai/v1"),
                "models": [
                    os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507"),  # 235B MoE
                    "llama3.1-8b",
                ],
            })

        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            self._alt_providers.append({
                "name": "Gemini",
                "client": _OAI(
                    api_key=gemini_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                "models": [
                    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),  # most capable
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ],
            })

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _is_retryable(self, err: str) -> bool:
        e = err.lower()
        return (
            "rate_limit_exceeded" in e or "429" in err
            or "413" in err
            or "too large" in e
            or "reduce your message size" in e
            or "not supported" in e
            or "failed to template" in e
            or "404" in err
        )

    def _is_unrecoverable(self, err: str) -> bool:
        e = err.lower()
        return any(t in e for t in ("invalid api key", "invalid_api_key", "authentication", "unauthorized", "forbidden"))

    def _trim_messages(self, kwargs: Dict[str, Any], label: str) -> None:
        msgs = kwargs["messages"]
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        kwargs["messages"] = system_msgs + non_system[-6:]
        print(f"[LLM] Trimmed history to last 6 messages for {label}.")

    def _build_result(self, msg, model: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"content": _strip_thinking(msg.content or ""), "model": model}
        tcs = _extract_tool_calls(msg)
        if tcs:
            result["tool_calls"] = tcs
            print(f"[LLM] {len(tcs)} tool call(s) returned.")
        return result

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        formatted: List[Dict[str, Any]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend(self._clean_messages(messages))

        kwargs: Dict[str, Any] = {"model": self.MODEL, "messages": formatted}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            print(f"[LLM] Tools available: {len(tools)}")

        print(f"[LLM] Sending request (messages: {len(messages)})")

        # ── Groq chain ────────────────────────────────────────────────────────
        last_error = ""
        tried: set = set()
        for model in self.GROQ_CHAIN:
            if model in tried:
                continue
            tried.add(model)
            kwargs["model"] = model
            if model in self.NO_TOOLS_MODELS and tools:
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                print(f"[LLM] Tools disabled for small model {model}.")
            elif tools and "tools" not in kwargs:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            is_first = model == self.GROQ_CHAIN[0]
            try:
                response = self.groq.chat.completions.create(**kwargs)
                print(f"[LLM] {'Response' if is_first else f'Fallback {model}'} received.")
                return self._build_result(response.choices[0].message, model)
            except Exception as e:
                last_error = str(e)
                is_too_large = "413" in last_error or "too large" in last_error.lower() or "reduce your message size" in last_error.lower()
                is_rate_limit = "429" in last_error or "rate_limit" in last_error
                if is_rate_limit:
                    print(f"[LLM] Rate limit on {model}. Trying next...")
                elif is_too_large:
                    print(f"[LLM] Request too large for {model}. Trimming and trying next...")
                    self._trim_messages(kwargs, model)
                else:
                    print(f"[LLM] {model} error: {last_error[:100]}. Trying next...")
                if self._is_unrecoverable(last_error):
                    break
                if not self._is_retryable(last_error):
                    print(f"[LLM] Non-retryable error from Groq: {last_error[:120]}")
                    break

        # ── Alternative providers ─────────────────────────────────────────────
        for provider in self._alt_providers:
            pname = provider["name"]
            pclient = provider["client"]
            for model in provider["models"]:
                alt_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": kwargs["messages"],
                }
                if tools and model not in self.NO_TOOLS_MODELS:
                    alt_kwargs["tools"] = tools
                    alt_kwargs["tool_choice"] = "auto"
                elif model in self.NO_TOOLS_MODELS and tools:
                    print(f"[LLM] Tools disabled for small model {model}.")
                try:
                    print(f"[LLM] Trying {pname} / {model}...")
                    response = pclient.chat.completions.create(**alt_kwargs)
                    print(f"[LLM] {pname} / {model} responded.")
                    return self._build_result(response.choices[0].message, f"{pname}/{model}")
                except Exception as e:
                    last_error = str(e)
                    print(f"[LLM] {pname}/{model} failed: {last_error[:120]}")
                    if self._is_unrecoverable(last_error):
                        break

        retry_msg = self._parse_retry_time(last_error)
        print(f"[LLM] All providers exhausted. {retry_msg}")
        return {
            "content": (
                f"All systems are currently throttled, sir. "
                f"{retry_msg} I'll be fully operational again shortly."
            ),
            "model": "none",
        }

    def _parse_retry_time(self, error_str: str) -> str:
        match = re.search(r"Please try again in ([^\.']+)", error_str)
        if match:
            return f"Groq requests a {match.group(1).strip()} cooldown."
        return "Retry time unknown."

    def _stream_chunks(self, stream) -> Generator[str, None, None]:
        """Yield stream chunks, suppressing any <think>...</think> blocks."""
        buf = ""
        in_think = False
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            buf += content
            while True:
                if in_think:
                    end = buf.find("</think>")
                    if end == -1:
                        buf = ""  # discard — still inside thinking block
                        break
                    buf = buf[end + len("</think>"):]
                    in_think = False
                else:
                    start = buf.find("<think>")
                    if start == -1:
                        yield buf
                        buf = ""
                        break
                    if start > 0:
                        yield buf[:start]
                    buf = buf[start + len("<think>"):]
                    in_think = True

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> Generator[str, None, None]:
        """Streaming call — no tool calls. Falls through all providers."""
        formatted: List[Dict[str, Any]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        formatted.extend(self._clean_messages(messages))

        last_error = ""

        # Groq
        for model in self.GROQ_CHAIN:
            try:
                print(f"[LLM] Streaming via Groq/{model}...")
                stream = self.groq.chat.completions.create(
                    model=model, messages=formatted, stream=True
                )
                yield from self._stream_chunks(stream)
                print("[LLM] Streaming complete.")
                return
            except Exception as e:
                last_error = str(e)
                if self._is_unrecoverable(last_error):
                    yield "Authentication error, sir. Please check the API key."
                    return
                print(f"[LLM] Groq/{model} stream failed: {last_error[:100]} — trying next...")

        # Alternative providers
        for provider in self._alt_providers:
            pname = provider["name"]
            pclient = provider["client"]
            for model in provider["models"]:
                try:
                    print(f"[LLM] Streaming via {pname}/{model}...")
                    stream = pclient.chat.completions.create(
                        model=model, messages=formatted, stream=True
                    )
                    yield from self._stream_chunks(stream)
                    print(f"[LLM] {pname}/{model} streaming complete.")
                    return
                except Exception as e:
                    last_error = str(e)
                    if self._is_unrecoverable(last_error):
                        yield "Authentication error, sir. Please check the API key."
                        return
                    print(f"[LLM] {pname}/{model} stream failed: {last_error[:100]} — trying next...")

        retry_msg = self._parse_retry_time(last_error)
        print(f"[LLM] All streaming providers exhausted. {retry_msg}")
        yield f"All systems are currently throttled, sir. {retry_msg} I'll be back online shortly."
