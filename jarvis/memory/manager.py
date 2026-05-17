"""
Memory Manager for J.A.R.V.I.S.

Two-layer memory system:
1. ChromaDB (vector store) — semantic search over past conversations
2. SQLite — structured facts (name, location, preferences)

No external API calls. All data stored locally.
"""

import os
import chromadb
import sqlite3
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from typing import List, Dict, Optional

load_dotenv()


class MemoryManager:
    """
    Manages both semantic and structured memory for Jarvis.

    Two layers:
    1. Vector memory (ChromaDB): Embeds past conversations, allows
       semantic search ("what did I ask about weather last week?")
    2. Structured facts (SQLite): Key-value facts about the user
       (name, location, preferences)

    Free tools:
    - ChromaDB: local vector DB, no API key needed
    - all-MiniLM-L6-v2: fast, free embedding model (~80MB)

    Example:
        memory = MemoryManager()

        # Save a conversation
        memory.save_conversation(
            user_msg="Set a reminder for tomorrow",
            assistant_msg="Reminder set for tomorrow at 9am"
        )

        # Store facts
        memory.set_fact("user_name", "Alice")
        memory.set_fact("user_location", "Kigali")

        # Retrieve facts
        name = memory.get_fact("user_name")  # "Alice"

        # Search past conversations
        results = memory.retrieve_relevant("reminders", n_results=3)

        # Build context for LLM
        context = memory.build_context("remind me of something")
    """

    def __init__(self, db_path: str = "") -> None:
        db_path = db_path or os.getenv("JARVIS_MEMORY_PATH", "./jarvis_memory")
        """
        Initialize the memory system.

        Sets up ChromaDB for vector storage and SQLite for structured facts.
        Creates directories and tables if they don't exist.

        Args:
            db_path: Path to the memory database directory
                    (will be created if it doesn't exist)

        Raises:
            Exception: If ChromaDB or SQLite initialization fails
        """
        try:
            print(f"[Memory] Initializing memory system at: {db_path}")

            # Initialize ChromaDB with built-in ONNX embedding (no extra deps)
            self.chroma: chromadb.PersistentClient = (
                chromadb.PersistentClient(path=db_path)
            )
            self._ef = ONNXMiniLM_L6_V2()
            self.collection = self.chroma.get_or_create_collection(
                name="conversations",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[Memory] ChromaDB collection initialized (all-MiniLM-L6-v2).")

            # Initialize SQLite for structured facts
            self.sql_conn: sqlite3.Connection = sqlite3.connect(
                f"{db_path}/facts.db", check_same_thread=False
            )
            self._init_sql()
            print(f"[Memory] SQLite database initialized.")
            print(f"[Memory] Memory system ready.")
        except Exception as e:
            print(f"[Memory] Error in __init__: {e}")
            raise

    def _init_sql(self) -> None:
        """
        Initialize SQLite tables for structured facts and metadata.

        Creates two tables:
        1. facts: key-value store for user properties
        2. conversations: metadata about saved conversations

        Returns:
            None
        """
        try:
            # Create facts table
            self.sql_conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create conversations metadata table
            self.sql_conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    summary TEXT,
                    timestamp TEXT NOT NULL,
                    raw_exchange TEXT
                )
            """)

            self.sql_conn.commit()
            print(f"[Memory] SQL tables initialized.")
        except Exception as e:
            print(f"[Memory] Error in _init_sql: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Vector Memory (ChromaDB)
    # ─────────────────────────────────────────────────────────────────────

    def save_conversation(self, user_msg: str, assistant_msg: str) -> None:
        """
        Save a conversation turn to vector memory.

        Embeds the conversation pair and stores in ChromaDB for
        semantic search. Allows retrieval of relevant past conversations.

        Args:
            user_msg: The user's message (str)
            assistant_msg: Jarvis's response (str)

        Returns:
            None

        Raises:
            Exception: On embedding or database error (caught internally)
        """
        try:
            if not user_msg or not assistant_msg:
                print(f"[Memory] Warning: Empty message in save_conversation")
                return

            text: str = f"User: {user_msg}\nJarvis: {assistant_msg}"
            doc_id: str = str(uuid.uuid4())

            # chromadb embeds automatically via the collection's embedding_function
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "user_msg": user_msg[:200],
                }]
            )

            print(f"[Memory] Conversation saved (ID: {doc_id[:8]}...)")
        except Exception as e:
            print(f"[Memory] Error in save_conversation: {e}")

    def retrieve_relevant(
        self,
        query: str,
        n_results: int = 5
    ) -> List[str]:
        """
        Retrieve semantically similar past conversations.

        Embeds the query and searches for similar conversations in ChromaDB.
        Useful for providing context to the LLM about past interactions.

        Args:
            query: The search query (natural language string)
            n_results: Number of results to return (default: 5)

        Returns:
            List of strings (past conversations), or empty list if none found

        Raises:
            Exception: On embedding or search error (caught internally)
        """
        try:
            # If no conversations stored yet, return empty
            if self.collection.count() == 0:
                return []

            # chromadb embeds the query automatically
            results: Dict[str, List] = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )

            # Extract and return documents
            documents: List[str] = results.get("documents", [[]])[0]
            print(f"[Memory] Retrieved {len(documents)} relevant conversations.")
            return documents
        except Exception as e:
            print(f"[Memory] Error in retrieve_relevant: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────
    # Structured Facts (SQLite)
    # ─────────────────────────────────────────────────────────────────────

    def set_fact(self, key: str, value: str) -> None:
        """
        Store or update a structured fact about the user.

        Examples:
        - set_fact("user_name", "Alice")
        - set_fact("user_location", "Kigali")
        - set_fact("preferred_language", "English")

        Args:
            key: Unique identifier for the fact (str)
            value: The fact value (str)

        Returns:
            None

        Raises:
            Exception: On database error (caught internally)
        """
        try:
            if not key or not value:
                print(f"[Memory] Warning: Empty key or value in set_fact")
                return

            self.sql_conn.execute("""
                INSERT OR REPLACE INTO facts (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))
            self.sql_conn.commit()

            print(f"[Memory] Fact stored: {key} = {value[:50]}")
        except Exception as e:
            print(f"[Memory] Error in set_fact: {e}")

    def get_fact(self, key: str) -> Optional[str]:
        """
        Retrieve a stored fact by key.

        Args:
            key: The fact key to look up

        Returns:
            The fact value (str), or None if not found

        Raises:
            Exception: On database error (caught internally, returns None)
        """
        try:
            cursor: sqlite3.Cursor = self.sql_conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            )
            row: Optional[tuple] = cursor.fetchone()
            if row:
                print(f"[Memory] Fact retrieved: {key}")
                return row[0]
            else:
                print(f"[Memory] Fact not found: {key}")
                return None
        except Exception as e:
            print(f"[Memory] Error in get_fact: {e}")
            return None

    def get_all_facts(self) -> Dict[str, str]:
        """
        Retrieve all stored facts as a dictionary.

        Returns:
            Dict mapping fact keys to values. Empty dict if no facts stored.

        Raises:
            Exception: On database error (caught internally, returns empty dict)
        """
        try:
            cursor: sqlite3.Cursor = self.sql_conn.execute(
                "SELECT key, value FROM facts"
            )
            facts: Dict[str, str] = {row[0]: row[1] for row in cursor.fetchall()}
            print(f"[Memory] Retrieved all facts ({len(facts)} total).")
            return facts
        except Exception as e:
            print(f"[Memory] Error in get_all_facts: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────
    # Context Building
    # ─────────────────────────────────────────────────────────────────────

    def build_context(self, query: str) -> str:
        """
        Build a memory context string to inject into the LLM prompt.

        Combines relevant past conversations and key facts into a single
        <memory> block that gets prepended to the user's message.

        This allows the LLM to respond with user context and history.

        Args:
            query: The user's current query (used to find relevant past conversations)

        Returns:
            String in <memory>...</memory> tags, or empty string if no context

        Example output:
            <memory>
            Known facts about you:
            - user_name: Alice
            - user_location: Kigali

            Relevant past conversations:
            User: Set a reminder for tomorrow
            Jarvis: Reminder set for 9am tomorrow

            ---

            User: What's the weather?
            Jarvis: It's 24°C and sunny in Kigali
            </memory>

        Raises:
            Exception: On retrieval error (caught internally, returns empty string)
        """
        try:
            context_parts: List[str] = []

            # Get all stored facts
            facts: Dict[str, str] = self.get_all_facts()
            if facts:
                facts_str: str = "\n".join(
                    f"- {k}: {v}" for k, v in facts.items()
                )
                context_parts.append(f"Known facts about you:\n{facts_str}")

            # Get relevant past conversations
            relevant_convos: List[str] = self.retrieve_relevant(
                query, n_results=3
            )
            if relevant_convos:
                convos_str: str = "\n---\n".join(relevant_convos)
                context_parts.append(
                    f"Relevant past conversations:\n{convos_str}"
                )

            # Combine and wrap in memory tags
            if not context_parts:
                return ""

            full_context: str = "\n\n".join(context_parts)
            result: str = f"<memory>\n{full_context}\n</memory>"

            print(f"[Memory] Context built ({len(context_parts)} sections).")
            return result
        except Exception as e:
            print(f"[Memory] Error in build_context: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Close database connections gracefully.

        Call this when shutting down Jarvis to ensure all data is flushed.

        Returns:
            None

        Raises:
            Exception: On database close error (caught internally)
        """
        try:
            if self.sql_conn:
                self.sql_conn.close()
            print(f"[Memory] Memory system closed.")
        except Exception as e:
            print(f"[Memory] Error in close: {e}")
