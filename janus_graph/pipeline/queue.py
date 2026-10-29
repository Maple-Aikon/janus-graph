"""Async SQLite WAL-backed persistent queue for episodes and DLQ."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import resolve_home_relative

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
    updated_at TEXT NOT NULL,
    last_replay_at TEXT
);
-- Indexes are created idempotently in _init_db_sync via _ensure_index_sync
-- so we can introduce new indexes without forcing ALTER + re-CREATE INDEX
-- on existing tables where the column has just been added.

CREATE TABLE IF NOT EXISTS dead_letter (
    episode_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    last_error TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    failed_at TEXT NOT NULL,
    recovered_at TEXT,
    recovered_attempts INTEGER
);

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


# ─── shared retry policy ──────────────────────────────────────────────────
#
# ONE policy, THREE write paths. Before this, each path had its own blind
# spot (measured 2026-10-23): the reaper had no filter at all, the DLQ batch
# replay used a narrow allowlist, and the manual single-episode replay had no
# filter AND reset ``attempt_count`` to 0 — so a manual retry could re-arm a
# permanent upstream fault and make the cycle repeatable by hand.
#
# The policy is a DENYLIST on measured PERMANENT upstream codes, not an
# allowlist of error classes. An allowlist silently starves every transient
# class it does not name; a denylist fails SAFE (an unknown error is retried,
# and an unknown *permanent* error costs one wasted round before it lands in
# the DLQ where ``dlq_open`` surfaces it).


# Mirrors ``config.RetryPolicyConfig``. Kept as a plain dataclass so the queue
# does not have to import the config module at call time (the CLI and the MCP
# server both build one from settings; tests build one directly).
@dataclass(frozen=True)
class RetryPolicy:
    """Shared requeue policy. See ``config.RetryPolicyConfig`` for the rationale."""

    permanent_error_codes: tuple[str, ...] = ("Error code: 404", "Error code: 402")
    allowed_error_codes: tuple[str, ...] = ()
    reap_limit: int = 24
    respect_attempt_ceiling: bool = True
    max_attempts: int = 3

    @classmethod
    def from_settings(cls, settings: Any) -> "RetryPolicy":
        """Build a policy from a settings object, defensively.

        ``getattr`` chains only: ``contracts.Settings`` has no ``pipeline``
        attribute at all, and ``JanusSettings.pipeline`` has no
        ``retry_policy`` until this section exists. A direct attribute chain
        would raise exactly when the guard was needed — the same defect class
        as ``cron.py:37-43`` and ``dream.py:116``.
        """
        pipeline = getattr(settings, "pipeline", None)
        policy = getattr(pipeline, "retry_policy", None)
        if policy is None:
            return cls()
        max_attempts = getattr(pipeline, "max_attempts", 3)
        return cls(
            permanent_error_codes=tuple(
                getattr(policy, "permanent_error_codes", None)
                or ("Error code: 404", "Error code: 402")
            ),
            allowed_error_codes=tuple(getattr(policy, "allowed_error_codes", None) or ()),
            reap_limit=int(getattr(policy, "reap_limit", 24)),
            respect_attempt_ceiling=bool(getattr(policy, "respect_attempt_ceiling", True)),
            max_attempts=int(max_attempts if max_attempts else 3),
        )


def _policy_where_clause(
    policy: Optional[RetryPolicy],
    column: str,
) -> tuple[List[str], List[Any]]:
    """Build the SQL predicate + params that EXCLUDE policy-forbidden rows.

    ``column`` is the error-text column (``episodes.last_error`` or
    ``dead_letter.last_error``).

    Two properties this function must preserve, both of which were bugs
    before it existed:

    1. **NULL-safe.** A bare ``last_error NOT LIKE ?`` evaluates to NULL for a
       NULL ``last_error``, and the WHERE clause then DROPS the row — the
       exact opposite of the policy's own rule ("a row with no ``last_error``
       IS requeued; do not drop rows you cannot classify"). The predicate is
       therefore written ``(col IS NULL OR col NOT LIKE ?)`` (or, when an
       allowlist is in force, ``col IS NOT NULL AND col LIKE ?``).

    2. **No LIKE wildcards from user data.** The configured codes are bound as
       parameters inside ``%...%``, so a ``_`` or ``%`` inside a configured
       code would be a wildcard (``LIKE '%_%'`` matches every non-empty
       string). The existing ``_dlq_class_predicate`` helper solved the same
       problem with ``instr(lower(col), lower(?)) > 0``; this uses the same
       approach, which has no wildcard semantics at all.

    Returns ``(clauses, params)``. ``clauses`` is empty when the policy
    imposes no restriction, and the caller then applies no predicate at all.
    """
    if policy is None:
        return [], []
    allowed = tuple(policy.allowed_error_codes or ())
    permanent = tuple(policy.permanent_error_codes or ())
    if allowed:
        # An explicit allowlist wins over the denylist. A row with a NULL
        # error cannot match an allowlist entry, but the policy says a row we
        # cannot classify must not be dropped — so it passes through.
        clauses = []
        params: List[Any] = []
        for code in allowed:
            clauses.append(f"({column} IS NULL OR instr(lower({column}), lower(?)) > 0)")
            params.append(code)
        return clauses, params
    if not permanent:
        return [], []
    clauses = []
    params = []
    for code in permanent:
        clauses.append(f"({column} IS NULL OR instr(lower({column}), lower(?)) = 0)")
        params.append(code)
    return clauses, params


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
        # NEVER ``Path(db_path).resolve()`` on a raw literal here: that
        # anchors a relative path to os.getcwd(). See
        # resolve_home_relative for the 2026-10-02 repro.
        self.db_path = Path(resolve_home_relative(db_path))
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
            # Idempotent migrations for upgrades from pre-replay schema.
            # Use PRAGMA table_info guard (not try/except) so we don't rely on
            # error-message strings and so the migration is robust to silent
            # partial states (column present, index missing).
            self._ensure_column_sync(conn, "episodes", "last_replay_at", "TEXT")
            # Legacy indexes (pre-2026-09 schema) and new replay/recovery indexes.
            for name, ddl in (
                ("idx_status", "CREATE INDEX idx_status ON episodes(status)"),
                ("idx_enqueued_at", "CREATE INDEX idx_enqueued_at ON episodes(enqueued_at)"),
                (
                    "idx_last_replay_at",
                    "CREATE INDEX idx_last_replay_at ON episodes(last_replay_at)",
                ),
                ("idx_dlq_failed_at", "CREATE INDEX idx_dlq_failed_at ON dead_letter(failed_at)"),
                (
                    "idx_dlq_recovered_at",
                    "CREATE INDEX idx_dlq_recovered_at ON dead_letter(recovered_at)",
                ),
            ):
                self._ensure_index_ddl_sync(conn, name, ddl)
            conn.commit()

    @staticmethod
    def _ensure_column_sync(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        col_type: str,
    ) -> None:
        """Add `column` to `table` if absent. Idempotent."""
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {r["name"] for r in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    @staticmethod
    def _ensure_index_ddl_sync(
        conn: sqlite3.Connection,
        index_name: str,
        ddl: str,
    ) -> None:
        """Create index if absent. Idempotent. ddl should be the full CREATE INDEX statement."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        ).fetchone()
        if row is None:
            conn.execute(ddl)

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
        """Mark an episode done.

        ``consecutive_failures`` is reset here so the counter means what its
        name says. It was a DEAD counter: written in exactly one place
        (``mark_failed``) and read in ZERO places in the whole repo (measured
        2026-10-23, ``rg`` over ``janus_graph/`` returns only schema/SELECT/
        INSERT plumbing). Resetting it changes no decision, and it costs one
        line. It is NOT reset on requeue by any other path, so a row that
        keeps failing keeps its streak visible.
        """
        now = _iso_now()

        def _sync_done():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE episodes
                    SET status = 'done', finished_at = ?, updated_at = ?,
                        consecutive_failures = 0
                    WHERE id = ?
                    """,
                    (now, now, episode_id),
                )
                # An episode can only reach the DLQ via mark_aborted, and a
                # later replay puts it back in the queue. When it finally
                # succeeds, its dead_letter row is stale: leaving recovered_at
                # NULL keeps it counted as "open" forever, which inflated
                # dlq_open in /health to ~5x the real backlog (measured
                # 2026-10-12: 272 of 341 "open" rows had status='done').
                # recovered_at keeps its meaning - the row really was
                # resolved - and recovered_attempts preserves the prior
                # attempt count for forensics.
                dlq_row = conn.execute(
                    "SELECT attempt_count FROM dead_letter"
                    " WHERE episode_id = ? AND recovered_at IS NULL",
                    (episode_id,),
                ).fetchone()
                if dlq_row is not None:
                    conn.execute(
                        "UPDATE dead_letter"
                        " SET recovered_at = ?, recovered_attempts = ?"
                        " WHERE episode_id = ? AND recovered_at IS NULL",
                        (now, dlq_row["attempt_count"], episode_id),
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
                cur = conn.execute(
                    "SELECT payload_json, attempt_count FROM episodes WHERE id = ?", (episode_id,)
                )
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

    async def reap_stuck_processing(self, timeout_sec: int = 900, max_attempts: int = 3) -> int:
        """Find processing records older than timeout and requeue or abort."""
        now = _iso_now()

        def _sync_reap_stuck():
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
                        # ``enqueued_at`` MUST be refreshed here too
                        # (2026-10-23, round-2 blocker): this path is a FOURTH
                        # writer of ``status='queued'`` and it runs on EVERY
                        # sweep tick, so leaving ``enqueued_at`` at its
                        # original value made it a live second feeder of the
                        # queue head — the same clause that closes the
                        # self-feeding reap loop.
                        conn.execute(
                            """
                            UPDATE episodes
                            SET status = 'queued', attempt_count = ?,
                                consecutive_failures = 0,
                                enqueued_at = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (attempts, now, now, ep_id),
                        )
                    reaped_count += 1
                conn.commit()
                return reaped_count

        async with self._write_lock:
            return await asyncio.to_thread(_sync_reap_stuck)

    async def reap_failed_or_aborted(
        self,
        limit: Optional[int] = None,
        policy: Optional[RetryPolicy] = None,
    ) -> int:
        """Requeue failed/aborted records back to queued for manual or cron retry.

        The signature default used to be ``limit: int = 500`` — a default that
        silently meant "reap 500 rows with no policy applied". That default
        WAS the bug (not just the call sites), so the default now comes from
        the shared policy: ``None`` means "use ``policy.reap_limit``".

        Four properties this method must hold, each of which was a measured
        defect (2026-10-23):

        1. **Ordered pick-set.** ``ORDER BY updated_at ASC`` — the reaper's
           own ``updated_at`` write makes the pick self-advancing, so a
           requeued row that fails again moves to the BACK of the queue
           instead of being picked again the next night.

        2. **``enqueued_at`` is refreshed.** THIS is the load-bearing half.
           Without it the reaped rows keep the oldest ``enqueued_at`` in the
           table, and ``claim_next_batch`` (which orders by ``enqueued_at
           ASC``) hands them straight back to the worker, which re-fails them,
           and the next night's reap picks them again — a self-feeding
           100-row loop (measured: the same 100 rows every night since
           2026-10-09; 5,228 of 5,328 failed/aborted rows never requeued).

        3. **The shared policy applies.** Rows whose ``last_error`` carries a
           permanent upstream code are never requeued by ANY path.

        4. **The attempt ceiling is respected.** A row that already exhausted
           its attempts is provably futile work; 60 of the 100 rows the
           reaper picked every night were already at ``attempt_count >= 3``.
        """
        now = _iso_now()
        eff_policy = policy or RetryPolicy()
        eff_limit = eff_policy.reap_limit if limit is None else int(limit)
        if eff_limit <= 0:
            return 0

        def _sync_reap_failed():
            with self._get_connection() as conn:
                where_parts: List[str] = ["status IN ('failed', 'aborted')"]
                params: List[Any] = []

                # (3) shared policy — NULL-safe, no LIKE wildcards.
                pol_clauses, pol_params = _policy_where_clause(eff_policy, "last_error")
                where_parts.extend(pol_clauses)
                params.extend(pol_params)

                # (4) attempt ceiling.
                if eff_policy.respect_attempt_ceiling:
                    where_parts.append("attempt_count < ?")
                    params.append(eff_policy.max_attempts)

                # The DLQ-skip applies to 'failed' rows ONLY. ``mark_aborted``
                # writes a dead_letter row with ``recovered_at IS NULL``, so
                # every aborted episode has an open DLQ row; a blanket
                # "skip rows with an open DLQ row" would exclude ALL aborted
                # rows. The DLQ row is the OWNERSHIP marker: a 'failed' row
                # with an open DLQ row is owned by ``replay_dlq_batch``,
                # while an 'aborted' row's DLQ row is its TERMINAL state.
                where_parts.append(
                    "NOT (status = 'failed' AND EXISTS ("
                    "SELECT 1 FROM dead_letter dl WHERE dl.episode_id = episodes.id"
                    " AND dl.recovered_at IS NULL))"
                )

                where_sql = " AND ".join(where_parts)
                cur = conn.execute(
                    f"""
                    SELECT id FROM episodes
                    WHERE {where_sql}
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    params + [eff_limit],
                )
                ids = [r["id"] for r in cur.fetchall()]
                if not ids:
                    return 0
                placeholders = ",".join("?" * len(ids))
                # (2) refresh ``enqueued_at`` — the clause that closes the
                # self-feeding loop. ``attempt_count`` is NOT reset (it is
                # the escalation signal read by ``should_retry``);
                # ``consecutive_failures`` IS reset, because it is written in
                # exactly one place and read in zero, so resetting it makes
                # the column mean what its name says.
                conn.execute(
                    f"UPDATE episodes SET status = 'queued', enqueued_at = ?, "
                    f"consecutive_failures = 0, updated_at = ? WHERE id IN ({placeholders})",
                    [now, now] + ids,
                )
                conn.commit()
                return len(ids)

        async with self._write_lock:
            return await asyncio.to_thread(_sync_reap_failed)

    async def replay_dlq_episode(
        self,
        episode_id: str,
        policy: Optional[RetryPolicy] = None,
    ) -> bool:
        """Replay a single DLQ episode.

        The shared policy applies here too (2026-10-23, round-2 blocker):
        this path used to write ``status='queued', attempt_count = 0,
        last_error = NULL`` for ONE row with NO class filter, NO denylist and
        NO ceiling — so a manual retry could re-arm a permanent upstream fault
        and reset its attempt counter, making the cycle repeatable by hand.
        Measured: 250 episodes carry ``last_replay_at`` and 27 of those were
        still failed/aborted with a permanent 404/402 code.

        A manual retry is still a manual retry: it must not be able to undo
        the policy that the automatic paths obey. A row whose error carries a
        permanent code is REFUSED (returns ``False``); the caller surfaces
        the refusal with the offending code named.

        ``attempt_count`` is still reset to 0 for the rows that ARE replayed —
        that is the semantic of a manual replay (the operator is asserting a
        fresh attempt), and the ceiling is re-evaluated from 0 on the next
        failure.
        """
        now = _iso_now()
        eff_policy = policy or RetryPolicy()

        def _sync_replay():
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT episode_id, attempt_count, last_error FROM dead_letter"
                    " WHERE episode_id = ?",
                    (episode_id,),
                )
                dlq_row = cur.fetchone()
                if not dlq_row:
                    return False
                # The policy reads the ERROR TEXT, which lives on the
                # dead_letter row. Apply it here so a permanent-code row is
                # refused BEFORE any write happens.
                pol_clauses, pol_params = _policy_where_clause(eff_policy, "last_error")
                if pol_clauses:
                    # The predicate is evaluated against the dead_letter row
                    # itself (column ``last_error``), not against episodes.
                    check = conn.execute(
                        "SELECT 1 FROM dead_letter WHERE episode_id = ? AND "
                        + " AND ".join(pol_clauses),
                        [episode_id] + pol_params,
                    ).fetchone()
                    if check is None:
                        return False
                conn.execute(
                    "UPDATE episodes SET status = 'queued', attempt_count = 0, last_error = NULL,"
                    " enqueued_at = ?, updated_at = ?, last_replay_at = ? WHERE id = ?",
                    (now, now, now, episode_id),
                )
                conn.execute(
                    "UPDATE dead_letter SET recovered_at = ?, recovered_attempts = ? WHERE episode_id = ?",
                    (now, dlq_row["attempt_count"], episode_id),
                )
                conn.commit()
                return True

        async with self._write_lock:
            return await asyncio.to_thread(_sync_replay)

    async def replay_dlq_batch(
        self,
        limit: int = 100,
        classes: Optional[List[str]] = None,
        min_age_sec: int = 0,
        policy: Optional[RetryPolicy] = None,
    ) -> int:
        """Replay up to ``limit`` DLQ rows back to queued, optionally filtered by error class.

        Args:
            limit: Maximum rows to replay in this call.
            classes: If provided, only replay rows whose ``last_error`` contains any
                of these substrings (case-insensitive LIKE match). E.g.
                ``["SCHEMA_DRIFT", "TIMEOUT"]`` replays only schema-drift and
                timeout errors, skipping budget-exceeded / 503 outages.
            min_age_sec: Skip rows whose ``failed_at`` is more recent than this
                many seconds ago. Set >0 to avoid re-replaying rows that just
                landed in DLQ (let one dream cycle confirm they really need replay).
            policy: The SHARED retry policy. The same denylist the reaper
                applies to ``episodes.last_error`` applies here to
                ``dead_letter.last_error``, so the two paths can no longer
                disagree about what is replayable. The ``classes`` allowlist
                stays as a NARROWER second filter; the denylist is the floor.

        Returns:
            Number of rows replayed.
        """
        now = _iso_now()
        eff_policy = policy or RetryPolicy()

        def _sync_replay_batch() -> int:
            with self._get_connection() as conn:
                where_parts = ["dl.recovered_at IS NULL"]
                params: List[Any] = []
                if classes:
                    class_clauses = " OR ".join(
                        ["LOWER(dl.last_error) LIKE LOWER(?)"] * len(classes)
                    )
                    where_parts.append(f"({class_clauses})")
                    params.extend(f"%{c}%" for c in classes)
                if min_age_sec > 0:
                    where_parts.append("datetime(dl.failed_at) <= datetime(?, ? || ' seconds')")
                    params.extend([now, f"-{min_age_sec}"])
                # Shared policy floor — same denylist, same NULL-safe
                # predicate, applied to the dead_letter row's error text.
                pol_clauses, pol_params = _policy_where_clause(eff_policy, "dl.last_error")
                if pol_clauses:
                    where_parts.append("(" + " AND ".join(pol_clauses) + ")")
                    params.extend(pol_params)
                where_sql = " AND ".join(where_parts)

                cur = conn.execute(
                    f"""
                    SELECT dl.episode_id, dl.attempt_count
                    FROM dead_letter dl
                    WHERE {where_sql}
                    ORDER BY dl.failed_at ASC
                    LIMIT ?
                    """,
                    params + [limit],
                )
                rows = cur.fetchall()
                if not rows:
                    return 0

                replayed = 0
                for row in rows:
                    ep_id = row["episode_id"]
                    prior_attempts = row["attempt_count"]
                    # Never requeue an episode that already succeeded. The
                    # episode table is the authority on whether the work is
                    # finished, not the dead_letter row - a replay that flips
                    # a 'done' episode back to 'queued' re-ingests content
                    # into the graph. measured 2026-10-12: mark_done now
                    # closes the DLQ row, so this can only be reached by a
                    # pre-existing stale row, but the guard is cheap.
                    cur = conn.execute(
                        "UPDATE episodes SET status = 'queued', attempt_count = 0,"
                        " last_error = NULL, enqueued_at = ?, updated_at = ?, last_replay_at = ?"
                        " WHERE id = ? AND status != 'done'",
                        (now, now, now, ep_id),
                    )
                    if cur.rowcount == 0:
                        # Episode is done (or gone): close the dead-letter row
                        # so it stops being counted as open, but do not claim
                        # it as replayed.
                        conn.execute(
                            "UPDATE dead_letter SET recovered_at = ?,"
                            " recovered_attempts = ? WHERE episode_id = ?",
                            (now, prior_attempts, ep_id),
                        )
                        continue
                    conn.execute(
                        "UPDATE dead_letter SET recovered_at = ?, recovered_attempts = ? WHERE episode_id = ?",
                        (now, prior_attempts, ep_id),
                    )
                    replayed += 1
                conn.commit()
                return replayed

        async with self._write_lock:
            return await asyncio.to_thread(_sync_replay_batch)

    async def record_dream_run(
        self,
        run_id: str,
        started_at: str,
        completed_at: Optional[str] = None,
        status: str = "running",
        episodes_evaluated: int = 0,
        phase_1_status: str = "SKIPPED",
        phase_2_status: str = "PENDING",
        phase_3_status: str = "PENDING",
        phase_4_status: str = "PENDING",
        error_message: Optional[str] = None,
    ) -> None:
        """INSERT one ``dream_runs`` row (the audit trail).

        The table already existed (``queue.py:49``) with every column this
        needs — but NOTHING ever inserted into it, so after 49 successful
        dream runs it had 0 rows. An empty table proves nothing unless a
        writer exists (2026-10-23, defect B5).

        Two calls per run: INSERT at start with ``status='running'`` and
        ``completed_at=NULL``, then UPDATE at the end with the four phase
        statuses. The INSERT is what makes the audit trail survive a FAILED
        run: the row is already there, so a crash mid-run still leaves
        evidence that the run started.
        """

        def _sync_record():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dream_runs (
                        run_id, started_at, completed_at, status, episodes_evaluated,
                        phase_1_status, phase_2_status, phase_3_status, phase_4_status,
                        error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        started_at,
                        completed_at,
                        status,
                        episodes_evaluated,
                        phase_1_status,
                        phase_2_status,
                        phase_3_status,
                        phase_4_status,
                        error_message,
                    ),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_record)

    async def update_dream_run(
        self,
        run_id: str,
        completed_at: Optional[str] = None,
        status: Optional[str] = None,
        episodes_evaluated: Optional[int] = None,
        phase_1_status: Optional[str] = None,
        phase_2_status: Optional[str] = None,
        phase_3_status: Optional[str] = None,
        phase_4_status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """UPDATE an existing ``dream_runs`` row (the end-of-run writer).

        Only the fields that are not ``None`` are written, so a partial
        update cannot blank out a column another caller set.
        """
        fields: List[str] = []
        params: List[Any] = []
        if completed_at is not None:
            fields.append("completed_at = ?")
            params.append(completed_at)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if episodes_evaluated is not None:
            fields.append("episodes_evaluated = ?")
            params.append(episodes_evaluated)
        if phase_1_status is not None:
            fields.append("phase_1_status = ?")
            params.append(phase_1_status)
        if phase_2_status is not None:
            fields.append("phase_2_status = ?")
            params.append(phase_2_status)
        if phase_3_status is not None:
            fields.append("phase_3_status = ?")
            params.append(phase_3_status)
        if phase_4_status is not None:
            fields.append("phase_4_status = ?")
            params.append(phase_4_status)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        if not fields:
            return

        def _sync_update():
            with self._get_connection() as conn:
                conn.execute(
                    f"UPDATE dream_runs SET {', '.join(fields)} WHERE run_id = ?",
                    params + [run_id],
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_update)

    def get_dream_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Read the most recent ``dream_runs`` rows, newest first."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT run_id, started_at, completed_at, status, episodes_evaluated,
                       phase_1_status, phase_2_status, phase_3_status, phase_4_status,
                       error_message
                FROM dream_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def reap_fairness_stats(self, max_attempts: int = 3) -> Dict[str, Any]:
        """Measure the reaper's FAIRNESS — how often it repeats the same rows.

        The whole point of the reap fix is fairness of turns: every eligible
        row should get its turn, exactly once, in order. Under the old code
        the repeat rate was **100 %** — the same 100 rows every night — and
        that single number would have exposed the self-feeding loop on day
        one.

        The signal is derived from the ROW STATE, not from a counter the
        reaper maintains, so it cannot drift from reality:

        * ``eligible`` — rows currently in failed/aborted that pass the
          attempt ceiling (the population the reaper draws from).
        * ``exhausted`` — rows in failed/aborted that are AT the ceiling and
          can therefore never be requeued (measured 2026-10-23: 60 of the
          100 nightly picks).
        * ``repeated_turns`` — eligible rows whose ``last_replay_at`` is set,
          i.e. rows that were already offered a turn before.
        * ``repeat_rate`` — ``repeated_turns / eligible`` (0.0 when eligible
          is 0, so the number is always defined).
        * ``oldest_enqueued_at`` / ``newest_enqueued_at`` — the queue-head
          spread. Under the old code every eligible row shared ONE
          ``enqueued_at`` (the oldest in the table), which is what made
          ``claim_next_batch`` hand them straight back.
        """
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT
                  COUNT(*) AS eligible,
                  SUM(CASE WHEN attempt_count >= ? THEN 1 ELSE 0 END) AS exhausted,
                  SUM(CASE WHEN last_replay_at IS NOT NULL THEN 1 ELSE 0 END) AS repeated
                FROM episodes
                WHERE status IN ('failed', 'aborted')
                """,
                (max_attempts,),
            )
            row = cur.fetchone()
            eligible = int(row["eligible"] or 0)
            exhausted = int(row["exhausted"] or 0)
            repeated = int(row["repeated"] or 0)
            head = conn.execute(
                """
                SELECT MIN(enqueued_at) AS oldest, MAX(enqueued_at) AS newest
                FROM episodes
                WHERE status IN ('failed', 'aborted')
                """
            ).fetchone()
        return {
            "eligible": eligible,
            "exhausted": exhausted,
            "repeated_turns": repeated,
            "repeat_rate": round(repeated / eligible, 4) if eligible else 0.0,
            "oldest_enqueued_at": head["oldest"] if head else None,
            "newest_enqueued_at": head["newest"] if head else None,
        }

    def dlq_refusal_reason(
        self,
        episode_id: str,
        policy: Optional[RetryPolicy] = None,
    ) -> Optional[str]:
        """Explain WHY a manual replay was refused, naming the offending code.

        Returns ``None`` when the episode is not in the DLQ at all (so the
        caller can say "not found"), and a human-readable string when the
        shared policy forbids the replay.

        The refusal must NAME the code, not just say "no" — an operator who
        is told "refused" learns nothing, while an operator who is told
        "refused: Error code: 404 (upstream model does not exist)" knows
        the row is inert and stops retrying it by hand.
        """
        eff_policy = policy or RetryPolicy()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT last_error FROM dead_letter WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        if row is None:
            return None
        last_error = row["last_error"] or ""
        allowed = tuple(eff_policy.allowed_error_codes or ())
        permanent = tuple(eff_policy.permanent_error_codes or ())
        lowered = last_error.lower()
        if allowed:
            for code in allowed:
                if code.lower() in lowered:
                    return None
            matched = None
        else:
            matched = None
            for code in permanent:
                if code.lower() in lowered:
                    matched = code
                    break
        if matched:
            return (
                "refused by the shared retry policy: the stored error carries the "
                f"permanent code {matched!r} ({last_error[:160]}). This is an "
                "upstream fault, not a transient one — retrying it cannot succeed."
            )
        return None

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
            dlq_cur = conn.execute(
                "SELECT COUNT(*) as count FROM dead_letter WHERE recovered_at IS NULL"
            )
            stats["dlq"] = dlq_cur.fetchone()["count"]
            return stats

    def count_total(self) -> int:
        """Return the total number of ``episodes`` rows, every status included.

        Kept separate from :meth:`get_stats` on purpose: that dict mixes the
        real statuses with the synthetic ``dlq`` key, so summing its values
        inflates the total by the open dead-letter count. No code path DELETEs
        from ``episodes``, so this number only ever grows.
        """
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) as count FROM episodes")
            return int(cur.fetchone()["count"])

    @staticmethod
    def _dlq_class_predicate(n: int) -> str:
        """SQL fragment: match ANY of ``n`` bound classes, case-insensitively.

        Kept as a ``@staticmethod`` because it reads nothing from the queue —
        a module function called through ``self`` would suggest instance state
        that does not exist.

        ``instr(lower(last_error), lower(?)) > 0`` instead of
        ``LIKE '%' || ? || '%'``: a caller-supplied ``_`` or ``%`` is a LIKE
        wildcard, and ``LIKE '%_%'`` matches every non-empty error (measured
        2026-10-16 on the live DB: 536/536 rows vs 375/536 for a literal
        underscore). Shared by :meth:`get_dlq_records` and
        :meth:`dlq_class_counts` so the two can never disagree on what a
        class means.

        The write path ``replay_dlq_batch`` keeps its ``LIKE`` predicate on
        purpose — changing which rows a replay selects is out of scope here.
        """
        return "(" + " OR ".join(["instr(lower(last_error), lower(?)) > 0"] * n) + ")"

    def get_dlq_records(
        self,
        limit: int = 20,
        classes: Optional[List[str]] = None,
        error_chars: Optional[int] = None,
        open_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Read ``dead_letter`` rows, newest ``failed_at`` first (2026-10-16).

        Every new argument defaults to "do not change the query", so
        ``get_dlq_records(limit=n)`` keeps the pre-change behaviour
        byte-for-byte: no WHERE clause, no truncation, recovered rows
        included. The 6 existing call sites (``mcp/server.py``, ``cli/main.py``,
        ``tests/test_queue*.py``, plus the ``list_dlq`` alias) keep working
        untouched.

        Args:
            limit: Max rows to return.
            classes: Substring match on ``last_error``, case-insensitive,
                OR-ed together. See :func:`_dlq_class_predicate` for why this
                is not ``LIKE``.
            error_chars: Truncate ``last_error`` to this many chars plus an
                ellipsis. ``None`` (default) and ``0`` both mean "no
                truncation" — 0 is the documented off-switch for the MCP tool,
                None is the API default that preserves existing output.
            open_only: ``True`` restricts to ``recovered_at IS NULL`` (the
                same predicate ``get_stats()['dlq']`` uses, so the counts
                cannot disagree). ``False`` selects only recovered rows.
                ``None`` (default) does not filter at all.
        """
        where_parts: List[str] = []
        params: List[Any] = []
        if open_only is not None:
            where_parts.append("recovered_at IS NULL" if open_only else "recovered_at IS NOT NULL")
        if classes:
            # Parens matter: without them an OR would escape the open_only
            # term and select recovered rows too.
            where_parts.append(self._dlq_class_predicate(len(classes)))
            params.extend(classes)
        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        with self._get_connection() as conn:
            cur = conn.execute(
                f"""
                SELECT episode_id, payload_json, last_error, attempt_count, failed_at, recovered_at
                FROM dead_letter
                {where_sql}
                ORDER BY failed_at DESC
                LIMIT ?
                """,
                params + [limit],
            )
            rows = [dict(row) for row in cur.fetchall()]

        if error_chars and error_chars > 0:
            for row in rows:
                err = row.get("last_error")
                if err and len(err) > error_chars:
                    row["last_error"] = err[:error_chars] + "…"
        return rows

    def dlq_class_counts(self, classes: Optional[List[str]]) -> Dict[str, int]:
        """Count OPEN dead-letter rows per class (2026-10-16).

        Single source of truth for the per-class breakdown: it reads the same
        ``recovered_at IS NULL`` predicate as :meth:`get_stats` and the same
        match predicate as :meth:`get_dlq_records`, so ``by_class`` cannot
        drift away from either the open total or the rows the filter returns.

        A row matching TWO classes counts in both — the counts are a
        breakdown of independent predicates, not a partition. Callers that
        need them to add up must compare the union against
        ``get_stats()['dlq']``; on the live DLQ the two taxonomy classes in
        ``config.yaml`` overlap on 0 rows (measured 2026-10-16), so they do
        add up there.

        Every requested class gets a key, value 0 when absent — a missing key
        is indistinguishable from a class nobody asked about, and a caller
        rendering a table needs both.

        Args:
            classes: Taxonomy to count against. ``None`` or ``[]`` returns
                ``{}``: with no taxonomy there is nothing to count, and
                returning "every row is unclassified" here would be a
                different claim than "the taxonomy matched nothing".
        """
        if not classes:
            return {}
        counts: Dict[str, int] = {}
        with self._get_connection() as conn:
            for cls in classes:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM dead_letter
                    WHERE recovered_at IS NULL
                      AND instr(lower(last_error), lower(?)) > 0
                    """,
                    (cls,),
                )
                counts[cls] = int(cur.fetchone()["count"])
        return counts

    def dlq_class_union_count(self, classes: Optional[List[str]]) -> int:
        """Count OPEN rows matching ANY of ``classes`` (2026-10-16).

        Distinct from summing :meth:`dlq_class_counts`: that double-counts a
        row whose error matches two classes, because the per-class counts are
        independent predicates rather than a partition. ``unclassified`` needs
        this number, not the sum.
        """
        if not classes:
            return 0
        with self._get_connection() as conn:
            cur = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM dead_letter
                WHERE recovered_at IS NULL
                  AND {self._dlq_class_predicate(len(classes))}
                """,
                list(classes),
            )
            return int(cur.fetchone()["count"])

    list_dlq = get_dlq_records
