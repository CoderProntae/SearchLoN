"""SQLite persistence for chat history and daily usage metrics."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import get_data_dir
from .limits import UsageLimits


def local_day() -> str:
    return datetime.now().astimezone().date().isoformat()


@dataclass(frozen=True)
class UsageSnapshot:
    day: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    active_seconds: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def remaining_requests(self, limits: UsageLimits) -> int:
        return max(0, limits.daily_requests - self.requests)

    def remaining_tokens(self, limits: UsageLimits) -> int:
        return max(0, limits.daily_token_limit - self.total_tokens)

    def remaining_searches(self, limits: UsageLimits) -> int:
        return max(0, limits.daily_search_limit - self.searches)


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    reason: str = ""
    day: str = ""


class DataStore:
    """Thread-safe, per-user history and usage store."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_data_dir() / "deepsearch.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._init_lock, closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session_idx
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS daily_usage (
                    day TEXT PRIMARY KEY,
                    requests INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    searches INTEGER NOT NULL DEFAULT 0,
                    active_seconds INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            connection.commit()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def create_session(self, title: str = "Yeni sohbet") -> str:
        session_id = str(uuid.uuid4())
        now = self._timestamp()
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title[:80], now, now),
            )
            connection.commit()
        return session_id

    def list_sessions(self, limit: int = 100) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Sohbet rolü user veya assistant olmalıdır.")
        now = self._timestamp()
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Sohbet bulunamadı: {session_id}")
            connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            title = exists["title"]
            if role == "user" and title == "Yeni sohbet":
                short_title = " ".join(content.strip().split())[:52] or "Yeni sohbet"
                connection.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (short_title, now, session_id),
                )
            else:
                connection.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
            connection.commit()

    def get_usage(self, day: str | None = None) -> UsageSnapshot:
        selected_day = day or local_day()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT day, requests, input_tokens, output_tokens, searches, active_seconds "
                "FROM daily_usage WHERE day = ?",
                (selected_day,),
            ).fetchone()
        if row is None:
            return UsageSnapshot(day=selected_day)
        return UsageSnapshot(**dict(row))

    def reserve_request(self, limits: UsageLimits, estimated_input_tokens: int = 0) -> QuotaDecision:
        """Atomically check daily limits and count one submitted request."""
        limits.validate()
        if estimated_input_tokens < 0:
            raise ValueError("Token tahmini negatif olamaz.")
        day = local_day()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_day(connection, day)
            row = self._usage_row(connection, day)
            if row["requests"] >= limits.daily_requests:
                connection.rollback()
                return QuotaDecision(False, "Günlük istek kotası doldu.", day)
            used_tokens = row["input_tokens"] + row["output_tokens"]
            if used_tokens + estimated_input_tokens >= limits.daily_token_limit:
                connection.rollback()
                return QuotaDecision(False, "Günlük token kotası bu isteği karşılamıyor.", day)
            if row["searches"] >= limits.daily_search_limit:
                connection.rollback()
                return QuotaDecision(False, "Günlük arama kotası doldu.", day)
            connection.execute(
                "UPDATE daily_usage SET requests = requests + 1 WHERE day = ?", (day,)
            )
            connection.commit()
        return QuotaDecision(True, day=day)

    def reserve_search(self, limits: UsageLimits, day: str | None = None) -> QuotaDecision:
        """Count a web search before it starts, including failed attempts."""
        limits.validate()
        selected_day = day or local_day()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_day(connection, selected_day)
            row = self._usage_row(connection, selected_day)
            if row["searches"] >= limits.daily_search_limit:
                connection.rollback()
                return QuotaDecision(False, "Günlük arama kotası doldu.", selected_day)
            connection.execute(
                "UPDATE daily_usage SET searches = searches + 1 WHERE day = ?", (selected_day,)
            )
            connection.commit()
        return QuotaDecision(True, day=selected_day)

    def record_tokens(self, input_tokens: int, output_tokens: int, day: str | None = None) -> None:
        selected_day = day or local_day()
        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        with closing(self._connect()) as connection:
            self._ensure_day(connection, selected_day)
            connection.execute(
                "UPDATE daily_usage SET input_tokens = input_tokens + ?, "
                "output_tokens = output_tokens + ? WHERE day = ?",
                (input_tokens, output_tokens, selected_day),
            )
            connection.commit()

    def record_active_seconds(self, seconds: float, day: str | None = None) -> None:
        selected_day = day or local_day()
        safe_seconds = max(0, round(seconds))
        with closing(self._connect()) as connection:
            self._ensure_day(connection, selected_day)
            connection.execute(
                "UPDATE daily_usage SET active_seconds = active_seconds + ? WHERE day = ?",
                (safe_seconds, selected_day),
            )
            connection.commit()

    @staticmethod
    def _usage_row(connection: sqlite3.Connection, day: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM daily_usage WHERE day = ?", (day,)).fetchone()
        assert row is not None
        return row

    @staticmethod
    def _ensure_day(connection: sqlite3.Connection, day: str) -> None:
        connection.execute("INSERT OR IGNORE INTO daily_usage(day) VALUES (?)", (day,))
