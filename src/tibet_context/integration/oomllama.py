"""
OomLlama memory bridge for tibet-context.

Reads conversation history from OomLlama's SQLite database
and converts it to ContextContainers. Can also inject context
back as prompt strings for model consumption.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tibet_core import Provider

from ..builder import ContextBuilder
from ..container import ContextContainer
from ..layers import CapabilityProfile, Layer, estimate_tokens
from ..reader import ContextReader
from ..gate import CapabilityGate

# Default OomLlama database path
DEFAULT_DB_PATH = "/srv/jtel-stack/data/chimera_memory.db"


@dataclass
class ConversationMessage:
    """A message from OomLlama's conversation memory."""
    role: str
    content: str
    provider: Optional[str]
    timestamp: str


@dataclass
class ConversationInfo:
    """Summary of a conversation."""
    id: str
    idd_name: str
    message_count: int
    created_at: str
    last_activity: str


class OomLlamaBridge:
    """
    Bridge between OomLlama's conversation memory and tibet-context.

    Reads from OomLlama's SQLite database (chimera_memory.db)
    and builds ContextContainers from conversation history.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        provider: Optional[Provider] = None,
        profile: Optional[CapabilityProfile] = None,
    ):
        self.db_path = db_path
        self.provider = provider or Provider(actor="tibet-context:oomllama")
        self.profile = profile or CapabilityProfile.default()
        self.builder = ContextBuilder(provider=self.provider, profile=self.profile)
        self.gate = CapabilityGate(profile=self.profile)
        self.reader = ContextReader(gate=self.gate)

    def _connect(self) -> sqlite3.Connection:
        """Connect to OomLlama's database."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"OomLlama database not found: {self.db_path}")
        return sqlite3.connect(self.db_path)

    def list_conversations(self, idd_name: Optional[str] = None) -> List[ConversationInfo]:
        """
        List conversations from OomLlama memory.

        Args:
            idd_name: Optional filter by IDD name

        Returns:
            List of conversation summaries
        """
        conn = self._connect()
        try:
            if idd_name:
                cursor = conn.execute("""
                    SELECT c.id, c.idd_name, COUNT(m.id), c.created_at, c.last_activity
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.id
                    WHERE c.idd_name = ?
                    GROUP BY c.id
                    ORDER BY c.last_activity DESC
                """, (idd_name,))
            else:
                cursor = conn.execute("""
                    SELECT c.id, c.idd_name, COUNT(m.id), c.created_at, c.last_activity
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.id
                    GROUP BY c.id
                    ORDER BY c.last_activity DESC
                """)

            return [
                ConversationInfo(
                    id=row[0], idd_name=row[1], message_count=row[2],
                    created_at=row[3], last_activity=row[4],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_messages(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """
        Get messages from a conversation.

        Args:
            conversation_id: UUID of the conversation
            max_messages: Optional limit on messages returned

        Returns:
            List of messages in chronological order
        """
        conn = self._connect()
        try:
            query = """
                SELECT role, content, provider, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
            """
            if max_messages:
                # Get the last N messages
                query = f"""
                    SELECT role, content, provider, timestamp FROM (
                        SELECT role, content, provider, timestamp, id
                        FROM messages
                        WHERE conversation_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                """
                cursor = conn.execute(query, (conversation_id, max_messages))
            else:
                cursor = conn.execute(query, (conversation_id,))

            return [
                ConversationMessage(
                    role=row[0], content=row[1],
                    provider=row[2], timestamp=row[3],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def from_conversation_memory(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        deep_context: Optional[str] = None,
    ) -> ContextContainer:
        """
        Build a ContextContainer from OomLlama conversation memory.

        Args:
            conversation_id: UUID of the conversation
            max_messages: Optional limit
            deep_context: Optional additional deep context for L2

        Returns:
            ContextContainer with conversation as layers
        """
        self.provider.create(
            action="oomllama.read_conversation",
            erin={"conversation_id": conversation_id},
            erachter="Reading conversation from OomLlama memory",
        )

        messages = self.get_messages(conversation_id, max_messages)
        if not messages:
            raise ValueError(f"No messages found for conversation: {conversation_id}")

        # Convert to builder format
        builder_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        return self.builder.from_conversation(
            builder_messages,
            session_id=conversation_id,
            deep_context=deep_context,
        )

    def inject_context(
        self,
        container: ContextContainer,
        model_id: str,
    ) -> str:
        """
        Generate a prompt injection string for a model.

        Formats the accessible layers as a context prefix
        that can be prepended to the model's prompt.

        Args:
            container: The context container
            model_id: Target model identifier

        Returns:
            Formatted context string for prompt injection
        """
        self.provider.create(
            action="oomllama.inject_context",
            erin={"container_id": container.id, "model_id": model_id},
            erachter=f"Injecting context for {model_id}",
        )

        content = self.reader.read(container, model_id)
        if not content:
            return ""

        return (
            f"<context source=\"tibet-context\" container=\"{container.id[:16]}\">\n"
            f"{content}\n"
            f"</context>\n\n"
        )

    def inject_context_for_capability(
        self,
        container: ContextContainer,
        capability: int,
    ) -> str:
        """
        Generate context string using raw capability value.

        Useful when model_id is not in the registry.
        """
        content = self.reader.read_for_capability(container, capability)
        if not content:
            return ""

        return (
            f"<context source=\"tibet-context\" container=\"{container.id[:16]}\">\n"
            f"{content}\n"
            f"</context>\n\n"
        )

    def stats(self) -> Dict[str, Any]:
        """Get memory stats from OomLlama database."""
        conn = self._connect()
        try:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            idd_count = conn.execute("SELECT COUNT(DISTINCT idd_name) FROM conversations").fetchone()[0]
            return {
                "total_conversations": conv_count,
                "total_messages": msg_count,
                "active_idds": idd_count,
                "db_path": self.db_path,
            }
        finally:
            conn.close()
