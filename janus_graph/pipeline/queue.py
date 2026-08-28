"""SQLite WAL-backed persistent queue for episodes and DLQ."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRecord:
    id: str
    status: str
    payload_json: str
    enqueued_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    attempt_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    checkpoint: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EpisodeQueue:
    """Manages episode storage, retrieval, checkpoints, and dead-letter queue."""

    def __init__(self, db_path: str = "./data/episodes.db"):
        self.db_path = Path(db_path).resolve()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'queued',
                    payload_json TEXT NOT NULL,
                    enqueued_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    checkpoint TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_status ON episodes(status);
                CREATE INDEX IF NOT EXISTS idx_enqueued_at ON episodes(enqueued_at);

                CREATE TABLE IF NOT EXISTS dead_letter (
                    episode_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    last_error TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    failed_at TEXT NOT NULL,
                    recovered_at TEXT,
                    recovered_attempts INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_dlq_failed_at ON dead_letter(failed_at);

                CREATE TABLE IF NOT EXISTS dream_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    episodes_evaluated INTEGER DEFAULT 0,
                    phase_1_status TEXT DEFAULT 'SKIPPED',
                    phase_2_status TEXT DEFAULT 'PENDING',
                    phase_3_status TEXT DEFAULT 'PENDING',
                    phase_4_status TEXT DEFAULT 'PENDING',
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS dream_entity_undo_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    source_node_uuid TEXT NOT NULL,
                    target_node_uuid TEXT NOT NULL,
                    source_node_snapshot TEXT NOT NULL,
                    edge_uuids_redirected TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES dream_runs(run_id)
                );
            """)

    def enqueue(self, content: str, group_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Enqueue an episode record."""
        ep_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps({"content": content, "group_id": group_id, "metadata": metadata or {}})
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO episodes (id, status, payload_json, enqueued_at, created_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (ep_id, payload, now, now, now),
            )
        return ep_id

    def get_stats(self) -> Dict[str, int]:
        """Return counts by status."""
        stats = {}
        with self._get_connection() as conn:
            for row in conn.execute("SELECT status, count(*) as c FROM episodes GROUP BY status"):
                stats[row["status"]] = row["c"]
            dlq_count = conn.execute("SELECT count(*) as c FROM dead_letter WHERE recovered_at IS NULL").fetchone()["c"]
            stats["dlq"] = dlq_count
        return stats
