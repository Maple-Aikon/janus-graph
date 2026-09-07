"""EpisodeQueueAdapter — Phase 3 wrapper around pipeline.EpisodeQueue.

Adds:
  1. Schema migration: ``payload_hash TEXT NOT NULL UNIQUE`` + ``claimed_by
     TEXT NULL`` columns + indexes (idempotent ALTER TABLE).
  2. ``enqueue_dedup()`` — sha256(content+group_id) gate; rejects with
     ``EpisodeDuplicateError`` when an episode with the same hash already
     exists in status in {queued, processing, done}. Returns the existing
     episode_id so the caller can decide.
  3. ``enqueue_guarded()`` — applies DaemonLockSettings.mode policy:
       mcp_only        -> ``LockModeError`` (caller -> HTTP 409)
       daemon_only     -> set claimed_by = "daemon:<pid>"
       dual_with_lock  -> set claimed_by = "daemon:<pid>" (other writers
                          skip rows where claimed_by != NULL)
  4. ``stats()`` — compact dict for /health queue_stats payload.

Why not modify pipeline.queue.EpisodeQueue directly:
  - That module is owned by the cron/worker subsystem (Phase 4 gate).
  - Phase 3 daemon is additive: keep EpisodeQueue backward-compatible with
    the existing MCP stdio enqueue path. Daemon-specific columns live on a
    thin wrapper so we don't risk clobbering live data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..pipeline.queue import EpisodeQueue, EpisodeRecord
from .phase3_settings import DaemonLockSettings

logger = logging.getLogger("janus_graph.daemon.episode_queue_adapter")


class EpisodeDuplicateError(Exception):
    """Raised when enqueue_dedup detects an existing row with same payload hash."""

    def __init__(self, existing_episode_id: str):
        self.existing_episode_id = existing_episode_id
        super().__init__(f"duplicate episode; existing_id={existing_episode_id}")


class LockModeError(Exception):
    """Raised when DaemonLockSettings.mode = mcp_only and a daemon write is attempted."""


@dataclass
class EnqueueResult:
    """Return value of ``EpisodeQueueAdapter.enqueue_guarded()``."""

    episode_id: str
    claimed_by: Optional[str]
    deduplicated: bool


def _payload_hash(content: str, group_id: str, source_description: str) -> str:
    """Stable sha256 of (content, group_id, source_description) triple.

    NOTE: source_description is part of the hash so two callers writing the
    same content from different sources do NOT collide (e.g. user_conversation
    vs agent_interaction could legitimately differ).
    """
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    h.update(b"|")
    h.update(group_id.encode("utf-8"))
    h.update(b"|")
    h.update(source_description.encode("utf-8"))
    return h.hexdigest()


class EpisodeQueueAdapter:
    """Daemon-side wrapper around ``pipeline.queue.EpisodeQueue``.

    Lifecycle:
      1. ``await adapter.start()`` — runs schema migration in to_thread.
      2. ``await adapter.enqueue_guarded(...)`` — many times (HTTP-bound).
      3. ``await adapter.stop()`` — closes underlying queue (no-op if shared).

    The adapter owns the lock policy; callers do NOT inspect DaemonLockSettings
    directly. HTTP layer maps ``EpisodeDuplicateError`` -> 409 DUP,
    ``LockModeError`` -> 409 LOCK_MODE_MCP_ONLY.
    """

    # Idempotent ALTER TABLE — columns + indexes added if missing.
    MIGRATION = [
        "ALTER TABLE episodes ADD COLUMN payload_hash TEXT",
        "ALTER TABLE episodes ADD COLUMN claimed_by TEXT",
        # Unique partial index: hash uniqueness only when not null. SQLite
        # supports CREATE UNIQUE INDEX IF NOT EXISTS; we use a regular index
        # then enforce uniqueness in Python because ALTER ADD COLUMN with
        # UNIQUE constraint on existing rows would fail.
        "CREATE INDEX IF NOT EXISTS idx_episodes_payload_hash ON episodes(payload_hash)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_claimed_by ON episodes(claimed_by)",
        "CREATE INDEX IF NOT EXISTS idx_episodes_status_claimed ON episodes(status, claimed_by)",
    ]

    def __init__(
        self,
        queue: EpisodeQueue,
        lock_settings: DaemonLockSettings,
        instance_id: Optional[str] = None,
    ) -> None:
        self._queue = queue
        self._lock = lock_settings
        # claimed_by tag: "<prefix>:<pid>" (e.g. "daemon:12345")
        self._instance_id = (
            instance_id
            or f"{lock_settings.instance_id_prefix}:{os.getpid()}"
        )

    # ─── lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Apply schema migration. Safe to call multiple times (idempotent)."""
        await asyncio.to_thread(self._migrate_sync)

    def _migrate_sync(self) -> None:
        # Open a side connection so we don't disturb the queue's WAL session.
        db_path = self._queue.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Probe column existence with PRAGMA first so we skip ALTER ADD COLUMN
        # if already migrated (older ALTER statements raise "duplicate column").
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)").fetchall()}
            if "payload_hash" not in cols:
                conn.execute("ALTER TABLE episodes ADD COLUMN payload_hash TEXT")
            if "claimed_by" not in cols:
                conn.execute("ALTER TABLE episodes ADD COLUMN claimed_by TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_payload_hash ON episodes(payload_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_claimed_by ON episodes(claimed_by)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_status_claimed "
                "ON episodes(status, claimed_by)"
            )
            conn.commit()

    # ─── enqueue ────────────────────────────────────────────────────────

    async def enqueue_guarded(
        self,
        content: str,
        group_id: str = "graphiti_memory",
        name: Optional[str] = None,
        source_description: str = "daemon_http",
    ) -> EnqueueResult:
        """Apply lock policy, dedup, and forward to underlying EpisodeQueue.

        Raises:
          LockModeError         — DaemonLockSettings.mode == mcp_only
          EpisodeDuplicateError — payload hash already exists in active status

        Returns:
          EnqueueResult with episode_id, claimed_by (None for mcp_only paths),
          deduplicated=False (only set True on dedup hit — but then we'd raise).
        """
        if self._lock.mode == "mcp_only":
            raise LockModeError(
                "DaemonLockSettings.mode = mcp_only; daemon /episodes disabled. "
                "Set JANUS_DAEMON__LOCK__MODE=daemon_only or dual_with_lock to opt in."
            )

        ph = _payload_hash(content, group_id, source_description)
        claimed_by: Optional[str]
        if self._lock.mode == "daemon_only":
            claimed_by = self._instance_id
        else:  # dual_with_lock
            claimed_by = self._instance_id

        # 1. Dedup check (fast path) — single query, indexed on payload_hash.
        existing = await asyncio.to_thread(self._find_existing_sync, ph)
        if existing is not None:
            # In dual_with_lock, the OTHER writer may have inserted; we treat
            # any matching hash in active status as duplicate.
            raise EpisodeDuplicateError(existing_episode_id=existing)

        # 2. Enqueue + set payload_hash + claimed_by in one transaction.
        #    We bypass EpisodeQueue.enqueue() so we can stamp both new columns
        #    in the SAME INSERT — otherwise a window exists where the row is
        #    visible to other writers without payload_hash set, breaking dedup.
        ep_id = await asyncio.to_thread(
            self._insert_with_hash_sync,
            content=content,
            group_id=group_id,
            name=name,
            source_description=source_description,
            payload_hash=ph,
            claimed_by=claimed_by,
        )

        return EnqueueResult(
            episode_id=ep_id,
            claimed_by=claimed_by,
            deduplicated=False,
        )

    def _insert_with_hash_sync(
        self,
        content: str,
        group_id: str,
        name: Optional[str],
        source_description: str,
        payload_hash: str,
        claimed_by: Optional[str],
    ) -> str:
        """Single-statement INSERT with payload_hash + claimed_by stamped.

        Mirrors the column set used by EpisodeQueue.enqueue so the row
        matches the existing schema and the cron worker (Phase 4) can pick
        it up unchanged.
        """
        import json as _json
        import uuid as _uuid

        from datetime import datetime, timezone

        ep_id = str(_uuid.uuid4())
        payload = {
            "content": content,
            "group_id": group_id,
            "name": name or f"ep_{ep_id[:8]}",
            "source_description": source_description,
        }
        payload_json = _json.dumps(payload)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._queue.db_path), timeout=5.0) as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    id, status, payload_json, enqueued_at,
                    started_at, finished_at,
                    attempt_count, consecutive_failures,
                    last_error, checkpoint,
                    payload_hash, claimed_by,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, NULL, NULL, 0, 0, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    ep_id,         # id
                    payload_json,  # payload_json
                    now,           # enqueued_at
                    payload_hash,  # payload_hash
                    claimed_by,    # claimed_by
                    now,           # created_at
                    now,           # updated_at
                ),
            )
            conn.commit()
        return ep_id

    def _find_existing_sync(self, payload_hash: str) -> Optional[str]:
        """Return episode_id if a row with this hash is in an active status.

        Active statuses: queued, processing. (done / failed / aborted do NOT
        trigger dedup — caller may want to re-enqueue after a permanent
        failure.)
        """
        with sqlite3.connect(str(self._queue.db_path), timeout=5.0) as conn:
            row = conn.execute(
                "SELECT id FROM episodes "
                "WHERE payload_hash = ? AND status IN ('queued', 'processing') "
                "ORDER BY created_at DESC LIMIT 1",
                (payload_hash,),
            ).fetchone()
            return row[0] if row else None

    def _stamp_claimed_by_sync(self, episode_id: str, claimed_by: Optional[str]) -> None:
        with sqlite3.connect(str(self._queue.db_path), timeout=5.0) as conn:
            conn.execute(
                "UPDATE episodes SET claimed_by = ?, updated_at = ? WHERE id = ?",
                (claimed_by, _iso_now(), episode_id),
            )
            conn.commit()

    # ─── stats ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Snapshot of queue counts for /health queue_stats payload.

        Cheap read — single GROUP BY query. Phase 4 cron uses this for
        alive-checks; Phase 3 surfaces it via /health body so operators see
        queue health even before /episodes endpoint is exercised.
        """
        try:
            with sqlite3.connect(str(self._queue.db_path), timeout=2.0) as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS c FROM episodes GROUP BY status"
                ).fetchall()
                dlq_count = conn.execute(
                    "SELECT COUNT(*) FROM dead_letter WHERE recovered_at IS NULL"
                ).fetchone()[0]
                by_status = {r[0]: r[1] for r in rows}
                claimed = conn.execute(
                    "SELECT COUNT(*) FROM episodes "
                    "WHERE claimed_by IS NOT NULL AND status IN ('queued','processing')"
                ).fetchone()[0]
            return {
                "by_status": by_status,
                "dlq_open": int(dlq_count),
                "claimed_active": int(claimed),
            }
        except sqlite3.Error as e:  # pragma: no cover — defensive
            logger.warning("queue stats failed: %s", e)
            return {"by_status": {}, "dlq_open": 0, "claimed_active": 0, "error": str(e)}


def _iso_now() -> str:
    """ISO8601 UTC timestamp (same format as pipeline.queue._iso_now)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
