"""SQLite persistence for the local chat application's conversations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStore:
    """Store chat turns and their retrieved source cards in a local SQLite file."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON conversation_messages(conversation_id, id);
                PRAGMA user_version = 1;
                """
            )

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations AS c
                LEFT JOIN conversation_messages AS m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(
        self,
        conversation_id: str,
        message_limit: int = 100,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                return None
            messages = connection.execute(
                """
                SELECT id, role, content, sources_json, created_at
                FROM (
                    SELECT id, role, content, sources_json, created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
                """,
                (conversation_id, max(1, min(message_limit, 500))),
            ).fetchall()

        result = dict(conversation)
        result["messages"] = [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in messages
        ]
        return result

    def append_turn(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, object]],
        conversation_id: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        conversation_id = conversation_id or str(uuid.uuid4())
        title = " ".join(question.split())[:72] or "New chat"

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (conversation_id, title, now, now),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

            connection.execute(
                "INSERT INTO conversation_messages "
                "(conversation_id, role, content, sources_json, created_at) "
                "VALUES (?, 'user', ?, '[]', ?)",
                (conversation_id, question, now),
            )
            connection.execute(
                "INSERT INTO conversation_messages "
                "(conversation_id, role, content, sources_json, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (conversation_id, answer, json.dumps(sources), now),
            )
        return conversation_id

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            return cursor.rowcount > 0
