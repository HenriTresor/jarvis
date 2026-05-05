"""
ReAct Agent Loop for J.A.R.V.I.S.

Implements the core reasoning engine: Reason → Act → Observe → Repeat.
Coordinates LLM brain, tools, and memory to handle complex user requests.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..brain.llm_client import LLMClient
from ..brain.prompts import JARVIS_SYSTEM_PROMPT
from ..memory.manager import MemoryManager
from .tools import TOOL_SCHEMAS
from .tool_executor import ToolExecutor


class JarvisAgent:
    """
    Core ReAct (Reason + Act) agent for Jarvis.

    Orchestrates the interaction loop:
    1. Retrieve memory context relevant to the user's query
    2. Send to LLM with available tools
    3. If LLM calls tools → execute them → feed results back to LLM
    4. Repeat until LLM produces a final text response
    5. Save conversation to long-term memory

    This implements the classic ReAct pattern from
    "ReAct: Synergizing Reasoning and Acting in Language Models"
    (Yao et al., 2023).

    Example:
        agent = JarvisAgent(location="Kigali, Rwanda")
        response = agent.think("What's the weather and my calendar for today?")
        print(response)  # "It's 24°C and sunny. You have 2 meetings today."
    """

    MAX_ITERATIONS: int = 8  # Prevent infinite loops

    def __init__(
        self,
        home_assistant_url: Optional[str] = None,
        ha_token: Optional[str] = None,
        location: str = "Kigali, Rwanda"
    ) -> None:
        """
        Initialize the Jarvis agent.

        Sets up the LLM brain, memory system, tool executor, and conversation history.

        Args:
            home_assistant_url: Home Assistant base URL (e.g., "http://localhost:8123")
            ha_token: Long-lived Home Assistant API token
            location: User's location for context (e.g., "Kigali, Rwanda")

        Raises:
            Exception: If LLMClient initialization fails (Ollama not running)
        """
        try:
            print(f"[Agent] Initializing Jarvis agent...")

            self.llm: LLMClient = LLMClient()
            self.memory: MemoryManager = MemoryManager()
            self.executor: ToolExecutor = ToolExecutor(
                home_assistant_url=home_assistant_url,
                ha_token=ha_token
            )
            self.location: str = location
            self.conversation_history: List[Dict[str, Any]] = []

            print(f"[Agent] Jarvis agent initialized.")
            print(f"[Agent] Location: {location}")
        except Exception as e:
            print(f"[Agent] Error in __init__: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt with current context injected.

        Formats the Jarvis system prompt with current datetime and location.

        Returns:
            Complete system prompt string ready for LLM

        Raises:
            Exception: On string formatting error (caught internally)
        """
        try:
            current_datetime: str = datetime.now().strftime(
                "%A, %B %d %Y at %I:%M %p"
            )
            prompt: str = JARVIS_SYSTEM_PROMPT.format(
                datetime=current_datetime,
                location=self.location
            )
            return prompt
        except Exception as e:
            print(f"[Agent] Error in _build_system_prompt: {e}")
            return JARVIS_SYSTEM_PROMPT

    def think(self, user_input: str) -> str:
        """
        Process a user input through the full ReAct loop.

        Main entry point: takes a user query, runs the reasoning + acting loop,
        and returns Jarvis's final text response.

        The loop:
        1. Retrieve memory context relevant to the query
        2. Send to LLM with memory + tools + system prompt
        3. If LLM returns tool calls:
           - Execute each tool
           - Add results back to conversation history
           - Send everything back to LLM for next turn
        4. Repeat until LLM returns a final text response (no tool calls)
        5. Save conversation to memory for future context

        Args:
            user_input: The user's message or command (str)

        Returns:
            Jarvis's final text response (str)

        Raises:
            Exception: On any internal error (caught, returns fallback response)
        """
        try:
            if not user_input or not user_input.strip():
                print(f"[Agent] Error: Empty user input")
                return "I didn't receive any input. Could you repeat that?"

            print(f"[Agent] Processing: '{user_input[:60]}...'")

            # ─────────────────────────────────────────────────────────────
            # Step 1: Retrieve memory context
            # ─────────────────────────────────────────────────────────────
            memory_context: str = self.memory.build_context(user_input)
            print(f"[Agent] Memory context built.")

            # ─────────────────────────────────────────────────────────────
            # Step 2: Build the user message with memory injected
            # ─────────────────────────────────────────────────────────────
            user_message_content: str = user_input
            if memory_context:
                user_message_content = f"{memory_context}\n\nUser: {user_input}"

            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message_content
            })

            # Build the system prompt with current context
            system_prompt: str = self._build_system_prompt()

            # ─────────────────────────────────────────────────────────────
            # Step 3: ReAct Loop
            # ─────────────────────────────────────────────────────────────
            iteration: int = 0

            while iteration < self.MAX_ITERATIONS:
                iteration += 1
                print(f"[Agent] Iteration {iteration}/{self.MAX_ITERATIONS}")

                # Call the LLM with tools
                response: Dict[str, Any] = self.llm.chat(
                    messages=self.conversation_history,
                    tools=TOOL_SCHEMAS,
                    system_prompt=system_prompt
                )

                # Check if LLM made tool calls
                tool_calls: List[Any] = response.get("tool_calls", [])

                if not tool_calls:
                    # ─────────────────────────────────────────────────────
                    # No more tool calls: this is the final response
                    # ─────────────────────────────────────────────────────
                    final_text: str = response.get("content", "").strip()
                    if not final_text:
                        final_text = "I'm not sure how to respond to that."

                    # Add assistant response to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_text
                    })

                    # Keep conversation history manageable (last 20 turns)
                    if len(self.conversation_history) > 20:
                        self.conversation_history = (
                            self.conversation_history[-20:]
                        )

                    # Save to long-term memory
                    self.memory.save_conversation(user_input, final_text)

                    print(f"[Agent] Interaction complete (iterations: {iteration})")
                    return final_text

                # ─────────────────────────────────────────────────────────
                # Tool calls detected: execute them and continue loop
                # ─────────────────────────────────────────────────────────
                print(f"[Agent] LLM made {len(tool_calls)} tool call(s)")

                # Add assistant response with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls
                })

                # Execute each tool call
                for tool_call in tool_calls:
                    try:
                        tool_name: str = tool_call["function"]["name"]
                        tool_args_str: str = tool_call["function"].get(
                            "arguments", "{}"
                        )

                        # Parse tool arguments
                        try:
                            tool_args: Dict[str, Any] = json.loads(
                                tool_args_str
                            )
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}

                        print(f"[Agent] Executing tool: {tool_name}")

                        # Execute the tool
                        result: str = self.executor.execute(
                            tool_name, tool_args
                        )

                        # Add tool result to conversation history
                        self.conversation_history.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get("id", "0"),
                            "tool_name": tool_name
                        })

                        print(f"[Agent] Tool result: {result[:100]}...")
                    except Exception as e:
                        print(f"[Agent] Error executing tool: {e}")
                        # Add error to history
                        self.conversation_history.append({
                            "role": "tool",
                            "content": f"Error executing tool: {e}",
                            "tool_call_id": tool_call.get("id", "0")
                        })

            # ─────────────────────────────────────────────────────────────
            # Max iterations reached without final response
            # ─────────────────────────────────────────────────────────────
            error_response: str = (
                "I ran into an issue processing that request. "
                "Could you rephrase or ask something simpler?"
            )
            self.conversation_history.append({
                "role": "assistant",
                "content": error_response
            })
            print(f"[Agent] Max iterations reached.")
            return error_response

        except Exception as e:
            print(f"[Agent] Error in think: {e}")
            return f"I encountered an error: {e}"

    def reset_conversation(self) -> None:
        """
        Reset the conversation history.

        Useful for starting fresh or between distinct conversational contexts.

        Returns:
            None
        """
        try:
            self.conversation_history = []
            print(f"[Agent] Conversation history reset.")
        except Exception as e:
            print(f"[Agent] Error in reset_conversation: {e}")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.

        Returns:
            List of message dicts (for debugging or analysis)

        Raises:
            Exception: On history retrieval error (caught internally)
        """
        try:
            return self.conversation_history.copy()
        except Exception as e:
            print(f"[Agent] Error in get_conversation_history: {e}")
            return []

    def close(self) -> None:
        """
        Clean up and close the agent gracefully.

        Closes memory database connections and other resources.

        Returns:
            None

        Raises:
            Exception: On cleanup error (caught internally)
        """
        try:
            self.memory.close()
            print(f"[Agent] Agent closed.")
        except Exception as e:
            print(f"[Agent] Error in close: {e}")
