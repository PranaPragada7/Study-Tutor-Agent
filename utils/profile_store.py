"""Transactional SQLite storage used by the local Study Tutor platform."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


class SQLiteProfileStore:
    """Small repository for versioned student-profile documents.

    The domain profile remains a JSON-compatible dictionary while SQLite adds
    transactional writes, safe concurrent access, indexed lookup, and a clear
    database boundary. This keeps the adaptive-learning code independent from
    a particular relational schema while the project is a single-user local
    application.
    """

    def __init__(self, data_dir: str, timeout_seconds: float = 10) -> None:
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "study_tutor.db")
        self.timeout_seconds = timeout_seconds

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(self.data_dir, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                profile_key TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        return connection

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield an immediate transaction so read/merge/write is atomic."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def read(self, profile_key: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM profiles WHERE profile_key = ?",
                (profile_key,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    @staticmethod
    def read_in_transaction(connection: sqlite3.Connection, profile_key: str) -> dict | None:
        row = connection.execute(
            "SELECT payload FROM profiles WHERE profile_key = ?",
            (profile_key,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    @staticmethod
    def upsert_in_transaction(
        connection: sqlite3.Connection,
        profile_key: str,
        display_name: str,
        payload: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO profiles (profile_key, display_name, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_key) DO UPDATE SET
                display_name = excluded.display_name,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                profile_key,
                display_name,
                payload,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def list_profiles(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT display_name, payload, updated_at FROM profiles "
                "ORDER BY lower(display_name), updated_at DESC"
            ).fetchall()

        profiles: list[dict] = []
        for row in rows:
            try:
                profile = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                continue
            profiles.append(
                {
                    "name": row["display_name"],
                    "total_quizzes": int(profile.get("total_quizzes", 0) or 0),
                    "courses": len(profile.get("courses", {}) or {}),
                    "updated_at": row["updated_at"],
                }
            )
        return profiles

    def delete(self, profile_key: str) -> bool:
        if not os.path.exists(self.path):
            return True
        with self._connect() as connection:
            connection.execute("DELETE FROM profiles WHERE profile_key = ?", (profile_key,))
        return True


__all__ = ["SQLiteProfileStore"]
