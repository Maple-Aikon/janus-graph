"""Async SQLite WAL-backed persistent queue for episodes and DLQ."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus_graph.pipeline.queue")

SCHEMA = """
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
"""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeRecord:
    id: str
    payload: dict[str, Any]
    status: str = "queued"
    enqueued_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    attempt_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    checkpoint: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def content(self) -> str:
        return self.payload.get("content", "")

    @property
    def group_id(self) -> str:
        return self.payload.get("group_id", "graphiti")

    @property
    def payload_json(self) -> str:
        return json.dumps(self.payload)


class EpisodeQueue:
    """Thread-safe and async-safe SQLite WAL queue manager."""

    def __init__(self, db_path: str = "./data/queue.db", busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path).resolve()
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: Optional[sqlite3.Connection] = None
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._init_db_sync()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_ms / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        return conn

    def _init_db_sync(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    async def enqueue(
        self,
        content: str,
        group_id: str = "graphiti",
        name: Optional[str] = None,
        source_description: str = "agent_interaction",
        episode_id: Optional[str] = None,
    ) -> str:
        """Enqueue a new episode payload into SQLite."""
        ep_id = episode_id or str(uuid.uuid4())
        payload = {
            "content": content,
            "group_id": group_id,
            "name": name or f"ep_{ep_id[:8]}",
            "source_description": source_description,
        }
        payload_json = json.dumps(payload)
        now = _iso_now()

        def _sync_insert():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO episodes (
                        id, status, payload_json, enqueued_at, attempt_count, 
                        consecutive_failures, created_at, updated_at
                    ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?)
                    """,
                    (ep_id, payload_json, now, now, now),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_insert)
        return ep_id

    async def claim_next_batch(self, limit: Optional[int] = 20) -> List[EpisodeRecord]:
        """Atomically claim next batch of queued episodes and set status to processing."""
        limit_val = limit if (limit is not None and limit > 0) else 20
        now = _iso_now()

        def _sync_claim():
            records: List[EpisodeRecord] = []
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, payload_json, status, enqueued_at, started_at, finished_at,
                           attempt_count, consecutive_failures, last_error, checkpoint
                    FROM episodes
                    WHERE status = 'queued'
                    ORDER BY enqueued_at ASC
                    LIMIT ?
                    """,
                    (limit_val,),
                )
                rows = cursor.fetchall()
                if not rows:
                    return records

                claimed_ids = [row["id"] for row in rows]
                placeholders = ",".join("?" * len(claimed_ids))
                conn.execute(
                    f"""
                    UPDATE episodes
                    SET status = 'processing', started_at = ?, updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now, now] + claimed_ids,
                )
                conn.commit()

                for row in rows:
                    try:
                        payload = json.loads(row["payload_json"])
                    except Exception:
                        payload = {"content": ""}
                    records.append(
                        EpisodeRecord(
                            id=row["id"],
                            payload=payload,
                            status="processing",
                            enqueued_at=row["enqueued_at"],
                            started_at=now,
                            finished_at=row["finished_at"],
                            attempt_count=row["attempt_count"],
                            consecutive_failures=row["consecutive_failures"],
                            last_error=row["last_error"],
                            checkpoint=row["checkpoint"],
                        )
                    )
            return records

        async with self._write_lock:
            return await asyncio.to_thread(_sync_claim)

    async def mark_done(self, episode_id: str) -> None:
        now = _iso_now()

        def _sync_done():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE episodes
                    SET status = 'done', finished_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, episode_id),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_done)

    async def mark_failed(self, episode_id: str, error: str) -> None:
        now = _iso_now()

        def _sync_failed():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE episodes
                    SET status = 'failed',
                        attempt_count = attempt_count + 1,
                        consecutive_failures = consecutive_failures + 1,
                        last_error = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (error, now, episode_id),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_failed)

    async def mark_aborted(self, episode_id: str, error: str) -> None:
        now = _iso_now()

        def _sync_aborted():
            with self._get_connection() as conn:
                # Fetch current record for DLQ
                cur = conn.execute("SELECT payload_json, attempt_count FROM episodes WHERE id = ?", (episode_id, ))
                row = cur.fetchone()
                payload_json = row["payload_json"] if row else "{}"
                attempts = (row["attempt_count"] if row else 0) + 1

                conn.execute(
                    """
                    UPDATE episodes
                    SET status = 'aborted',
                        attempt_count = ?,
                        last_error = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (attempts, error, now, episode_id),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dead_letter (\n                        episode_id, payload_json, last_error, attempt_count, failed_at\n                    ) VALUES (?, ?, ?, ?, ?)\n                    """,
                    (episode_id, payload_json, error, attempts, now),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_aborted)

    async def update_checkpoint(self, episode_id: str, checkpoint: str) -> None:
        now = _iso_now()

        def _sync_checkpoint():
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE episodes SET checkpoint = ?, updated_at = ? WHERE id = ?",
                    (checkpoint, now, episode_id),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_checkpoint)

    async def reap_stuck_processing(self, timeout_sec: int = 300, max_attempts: int = 3) -> int:
        """Find processing records older than timeout and requeue or abort."""
        now = _iso_now()

        def _sync_reap():
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT id, started_at, attempt_count, payload_json
                    FROM episodes
                    WHERE status = 'processing'
                    """
                )
                stuck_rows = []
                for row in cur.fetchall():
                    started_at = row["started_at"]
                    if not started_at:
                        continue
                    try:
                        dt = datetime.fromisoformat(started_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age = (datetime.now(timezone.utc) - dt).total_seconds()
                        if age >= timeout_sec:
                            stuck_rows.append(row)
                    except Exception:
                        stuck_rows.append(row)

                reaped_count = 0
                for row in stuck_rows:
                    ep_id = row["id"]
                    attempts = row["attempt_count"] + 1
                    if attempts >= max_attempts:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET status = 'aborted', attempt_count = ?, last_error = 'Processing timeout exceeded', updated_at = ?
                            WHERE id = ?
                            """,
                            (attempts, now, ep_id),
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO dead_letter (\n                                episode_id, payload_json, last_error, attempt_count, failed_at\n                            ) VALUES (?, ?, 'Processing timeout exceeded', ?, ?)\n                            """,
                            (ep_id, row["payload_json"], attempts, now),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE episodes
                            SET status = 'queued', attempt_count = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (attempts, now, ep_id),
                        )
                    reaped_count += 1
                conn.commit()
                return reaped_count

        async with self._write_lock:
            return await asyncio.to_thread(_sync_reap)

    async def reap_failed_or_aborted(self, limit: int = 500) -> int:
        """Requeue failed/aborted records back to queued for manual or cron retry."""
        now = _iso_now()

        def _sync_reap():
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT id FROM episodes
                    WHERE status IN ('failed', 'aborted')
                    LIMIT ?
                    """,
                    (limit,),
                )
                ids = [r["id"] for r in cur.fetchall()]
                if not ids:
                    return 0
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE episodes SET status = 'queued', updated_at = ? WHERE id IN ({placeholders})",
                    [now] + ids,
                )
                conn.commit()
                return len(ids)

        async with self._write_lock:
            return await asyncio.to_thread(_sync_reap)

    async def replay_dlq_episode(self, episode_id: str) -> bool:
        """Replay a single DLQ episode."""
        now = _iso_now()

        def _sync_replay():
            with self._get_connection() as conn:
                cur = conn.execute("SELECT episode_id, attempt_count FROM dead_letter WHERE episode_id = ?", (episode_id,))
                dlq_row = cur.fetchone()
                if not dlq_row:
                    return False
                conn.execute(
                    "UPDATE episodes SET status = 'queued', attempt_count = 0, last_error = NULL, updated_at = ? WHERE id = ?",
                    (now, episode_id),
                )
                conn.execute(
                    "UPDATE dead_letter SET recovered_at = ?, recovered_attempts = ? WHERE episode_id = ?",
                    (now, dlq_row["attempt_count"], episode_id),
                )
                conn.commit()
                return True

        async with self._write_lock:
            return await asyncio.to_thread(_sync_replay)

    def get_record(self, episode_id: str) -> Optional[EpisodeRecord]:
        """Fetch a single episode record by ID."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, payload_json, status, enqueued_at, started_at, finished_at,
                       attempt_count, consecutive_failures, last_error, checkpoint, created_at, updated_at
                FROM episodes
                WHERE id = ?
                """,
                (episode_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                payload = {}
            return EpisodeRecord(
                id=row["id"],
                payload=payload,
                status=row["status"],
                enqueued_at=row["enqueued_at"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                attempt_count=row["attempt_count"],
                consecutive_failures=row["consecutive_failures"],
                last_error=row["last_error"],
                checkpoint=row["checkpoint"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_stats(self) -> Dict[str, int]:
        """Return synchronous aggregate count of statuses."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT status, COUNT(*) as count FROM episodes GROUP BY status")
            stats = {row["status"]: row["count"] for row in cursor.fetchall()}
            dlq_cur = conn.execute("SELECT COUNT(*) as count FROM dead_letter WHERE recovered_at IS NULL")
            stats["dlq"] = dlq_cur.fetchone()["count"]
            return stats

    def get_dlq_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT episode_id, payload_json, last_error, attempt_count, failed_at, recovered_at
                FROM dead_letter
                ORDER BY failed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    list_dlq = get_dlq_records
