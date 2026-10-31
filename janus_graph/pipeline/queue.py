"""Async SQLite WAL-backed persistent queue for episodes and DLQ."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import resolve_home_relative
from ..core.dedup import (
    ACTIVE_STATUSES as _SHARED_ACTIVE_STATUSES,
)
from ..core.dedup import (
    BACKSTOP_SKIPPED_STATUS as _SHARED_BACKSTOP_SKIPPED_STATUS,
)
from ..core.dedup import (
    NON_BLOCKING_STATUSES as _SHARED_NON_BLOCKING_STATUSES,
)
from ..core.dedup import (
    UNIQUE_INDEX_DDL as _SHARED_UNIQUE_INDEX_DDL,
)
from ..core.dedup import (
    UNIQUE_INDEX_NAME as _SHARED_UNIQUE_INDEX_NAME,
)
from ..core.dedup import (
    canonical_payload_hash,
    find_active_duplicate,
    find_any_duplicate,
    recent_dedup_events,
    record_dedup_event,
    requeue_guarded,
)

logger = logging.getLogger("janus_graph.pipeline.queue")

#: The three dedup modes. ``off`` is the default and means "no behaviour
#: change at all" — see ``DedupModeSettings`` in config.py for the full
#: contract.
DEDUP_MODES = ("off", "observe", "enforce")

#: Fallback used when the config tree carries no usable value (e.g. a
#: queue constructed directly in a test, with no settings object). Default
#: is the SAFE mode: dedup off.
_DEFAULT_DEDUP_MODE = "off"

#: Cache for the config-resolved dedup mode. ``None`` = not resolved yet.
#: See ``_resolve_dedup_mode`` for why this exists.
_CONFIG_DEDUP_MODE: Optional[str] = None


class DuplicateEpisodeError(Exception):
    """Raised when enqueue rejects an episode as a duplicate.

    Carries the id of the row that already holds this payload, so the
    caller can report "already queued as <id>" instead of a bare failure.

    ``existing_episode_id`` is ``None`` only in the pathological case
    where the UNIQUE index rejected the INSERT and no holder could be
    found at all (the holder left the active set between the rejected
    INSERT and the lookup, AND no other row carries the hash). The
    message says so explicitly rather than pretending to know an id.
    """

    def __init__(self, existing_episode_id: Optional[str], payload_hash: Optional[str] = None):
        self.existing_episode_id = existing_episode_id
        self.payload_hash = payload_hash
        super().__init__(
            (
                f"duplicate episode; existing_id={existing_episode_id}"
                if existing_episode_id
                else "duplicate episode; existing id unknown"
            )
            + (f" hash={(payload_hash or '')[:12]}" if payload_hash else "")
        )


class DuplicateRaceError(Exception):
    """Raised internally when the UNIQUE index rejects an INSERT.

    This is the CROSS-PROCESS race the partial UNIQUE index exists to
    close: another writer inserted the same active payload between our
    pre-check SELECT and our INSERT.

    It is a distinct type from ``DuplicateEpisodeError`` because it is
    NOT a policy decision — the writer did not choose to reject, the
    database did. The enqueue path translates it into the policy
    decision (reject in ``enforce``, enqueue-and-record in ``observe``)
    so callers only ever see ``DuplicateEpisodeError``.

    It never escapes the queue: ``enqueue`` catches it and converts it.
    """

    def __init__(self, existing_episode_id: Optional[str], payload_hash: Optional[str] = None):
        self.existing_episode_id = existing_episode_id
        self.payload_hash = payload_hash
        super().__init__(
            "duplicate detected by the unique index"
            + (f"; existing_id={existing_episode_id}" if existing_episode_id else "")
        )


def _resolve_dedup_mode(dedup_mode: Optional[str] = None) -> str:
    """Resolve the effective dedup mode.

    Precedence:
      1. explicit ``dedup_mode`` argument (per-call override);
      2. ``cfg.pipeline.dedup.mode`` (YAML / env);
      3. ``"off"`` (safe default).

    An unrecognised value is NOT silently coerced to "off" — an unknown
    mode means the operator typo'd a config value, and quietly falling
    back would either disable a gate they enabled or enable one they
    didn't. It is logged and treated as off (safe), never as enforce.

    The config lookup is CACHED. ``load_config()`` re-reads and re-parses
    the YAML file on every call (there is no cache inside it), so calling
    it per-enqueue would add a file read + YAML parse to every MCP
    ``add_episode`` call — a real hot-path regression for a gate whose
    default is "off". The cache is process-local and invalidated by
    ``reset_dedup_mode_cache()``, which the tests use; a config change
    needs a process restart to take effect, which is the same contract
    the rest of the daemon already has (MCP stdio servers are spawned
    per parent process).
    """
    if dedup_mode is not None:
        mode = str(dedup_mode).strip().lower()
        if mode in DEDUP_MODES:
            return mode
        logger.warning(
            "unknown dedup_mode %r passed to enqueue; falling back to %r",
            dedup_mode,
            _DEFAULT_DEDUP_MODE,
        )
        return _DEFAULT_DEDUP_MODE

    global _CONFIG_DEDUP_MODE
    if _CONFIG_DEDUP_MODE is None:
        try:
            from ..config import load_config

            cfg = load_config()
            mode = str(cfg.pipeline.dedup.mode).strip().lower()
            if mode not in DEDUP_MODES:
                logger.warning(
                    "unknown cfg.pipeline.dedup.mode %r; falling back to %r",
                    mode,
                    _DEFAULT_DEDUP_MODE,
                )
                mode = _DEFAULT_DEDUP_MODE
            _CONFIG_DEDUP_MODE = mode
        except Exception:  # pragma: no cover - config load failure must not break enqueue
            logger.debug("could not resolve dedup mode from config", exc_info=True)
            _CONFIG_DEDUP_MODE = _DEFAULT_DEDUP_MODE
    return _CONFIG_DEDUP_MODE


def reset_dedup_mode_cache() -> None:
    """Drop the cached config-resolved dedup mode.

    Only useful for tests and for a process that re-reads its config
    without restarting. In production the MCP stdio server is respawned
    with its parent, so a restart is the normal way to pick up a config
    change.
    """
    global _CONFIG_DEDUP_MODE
    _CONFIG_DEDUP_MODE = None


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

-- ─── dedup ledger (2026-10-31) ────────────────────────────────────────
-- Durable record of every duplicate DETECTED by the enqueue gate.
--
-- WHY THIS TABLE EXISTS: before it, the dedup hit counter was a plain
-- Python attribute (``EpisodeQueue._dedup_hits``) living in the process
-- that happened to detect the duplicate. The PicoClaw hook spawns a FRESH
-- python process for every write, so the counter was always 0 when anybody
-- read it -- ``observed_hits`` was a number that LIES. The log line was
-- discarded for the same reason (the spawned process never called
-- ``logging.basicConfig``).
--
-- Writing the event to the DB makes the counter survive the process that
-- produced it, so the observe rollout can be judged from a real number.
--
-- ``detected_at`` is the ONLY timestamp: an event row is a point-in-time
-- observation, not a lifecycle. There is no "resolved" column because a
-- detected duplicate needs no resolution -- the row that caused it is
-- either still active or already terminal.
CREATE TABLE IF NOT EXISTS dedup_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_hash TEXT,
    existing_episode_id TEXT,
    duplicate_episode_id TEXT,
    writer TEXT
);

CREATE INDEX IF NOT EXISTS idx_dedup_events_detected_at
    ON dedup_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_dedup_events_mode
    ON dedup_events(mode);
"""

#: The status values the enqueue gate treats as "this payload is already in
#: flight" — a duplicate of one of these is rejected in ``enforce`` mode.
#:
#: Re-exported from ``core.dedup`` so there is ONE definition shared by
#: the Python-level check, the DB index predicate and the daemon writer.
#: The UNIQUE partial index below is scoped to exactly this set, so the DB
#: invariant and the Python check can never disagree about what "active"
#: means. If this tuple changes, the index predicate must change with it.
ACTIVE_STATUSES = _SHARED_ACTIVE_STATUSES

#: Terminal-ish statuses that do NOT block a new enqueue: re-queueing after
#: a permanent failure is a legitimate retry, and blocking it would strand
#: the row forever.
NON_BLOCKING_STATUSES = _SHARED_NON_BLOCKING_STATUSES

#: Name of the UNIQUE partial index that makes the dedup gate atomic.
#: Scoped to the ACTIVE statuses, so a row that has already been ingested
#: (status='done') does not block a later write of the same content — that
#: is exactly the "done-aware" freedom the enqueue gate has always had.
UNIQUE_INDEX_NAME = _SHARED_UNIQUE_INDEX_NAME

#: DDL for that index. The predicate must list ACTIVE_STATUSES literally —
#: SQLite does not accept a bind parameter in a partial index predicate.
UNIQUE_INDEX_DDL = _SHARED_UNIQUE_INDEX_DDL

#: The status a row is given when the processing-layer backstop decides the
#: payload was already ingested. It is deliberately NOT one of the failure
#: statuses: the row was not a failure, so it must not be reaped, retried or
#: dead-lettered. It is also NOT 'done' — that status means "this content
#: WAS ingested", and claiming it for a row that never was would inflate
#: ``dream.py``'s ``done + queued`` episode count with content the graph
#: never saw.
BACKSTOP_SKIPPED_STATUS = _SHARED_BACKSTOP_SKIPPED_STATUS


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
        # Observed duplicate count since process start. Incremented in both
        # observe and enforce mode. Exposed via get_stats() so the
        # observe-only rollout can be judged from a number instead of a
        # feeling.
        self._dedup_hits = 0
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
            # Dedup columns (1b). The daemon adapter migrates these too, but
            # a queue can be created against a fresh DB the daemon never
            # touched — and the MCP writer's INSERT references them.
            self._ensure_column_sync(conn, "episodes", "payload_hash", "TEXT")
            self._ensure_column_sync(conn, "episodes", "claimed_by", "TEXT")
            # Canonical hash (2026-10-31). ``payload_hash`` is the LEGACY
            # column: it was written by the daemon adapter and by the 1b
            # MCP path, but the 11 482 ``signet_*_reimport`` backfill rows
            # carry a hash from an UNKNOWN scheme (measured 2026-10-31: 367
            # rows match the canonical scheme, 11 482 do not, and none of
            # 6 plausible sha256 orderings reproduces them).
            #
            # So the legacy column is PRESERVED verbatim and a NEW column
            # carries the canonical hash. The backfill below only fills
            # ``canonical_hash`` for rows whose stored ``payload_hash``
            # EQUALS the recomputed canonical hash — i.e. rows we can PROVE
            # are canonical. Everything else keeps ``canonical_hash IS
            # NULL``, which the unique index treats as "unknown => never a
            # duplicate". Nothing is deleted or rewritten.
            self._ensure_column_sync(conn, "episodes", "canonical_hash", "TEXT")
            # Which gate wrote this row: off / observe / enforce. It is
            # bookkeeping AND it decides whether the UNIQUE partial index
            # covers the row — observe-mode rows must stay duplicate-able
            # (observe never rejects), enforce-mode rows must not.
            self._ensure_column_sync(conn, "episodes", "dedup_gate", "TEXT")
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
                (
                    "idx_episodes_payload_hash",
                    "CREATE INDEX idx_episodes_payload_hash ON episodes(payload_hash)",
                ),
                (
                    "idx_episodes_claimed_by",
                    "CREATE INDEX idx_episodes_claimed_by ON episodes(claimed_by)",
                ),
                (
                    "idx_episodes_canonical_hash",
                    "CREATE INDEX idx_episodes_canonical_hash ON episodes(canonical_hash)",
                ),
            ):
                self._ensure_index_ddl_sync(conn, name, ddl)
            self._backfill_canonical_hash_sync(conn)
            self._ensure_unique_active_hash_index_sync(conn)
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

    @staticmethod
    def _backfill_canonical_hash_sync(conn: sqlite3.Connection) -> None:
        """Fill ``canonical_hash`` for every row whose payload triple is complete.

        WHY ALL RECOMPUTABLE ROWS, NOT ONLY "PROVABLE" ONES
        -----------------------------------------------------
        The first cut of this backfill only filled rows whose stored
        ``payload_hash`` EQUALS the canonical hash recomputed from their
        own payload triple. The reasoning was that the 11 482
        ``signet_*_reimport`` rows carry a hash from an UNKNOWN scheme,
        so their canonical hash could not be trusted.

        That reasoning was too narrow, and being too narrow has a real
        cost: it left 12 345 of 12 712 rows with ``canonical_hash IS
        NULL``, which makes them INVISIBLE to the processing-layer
        backstop. The backstop exists to answer "was this exact content
        already ingested?" — and for the entire pre-migration
        population the answer would have been "no", which is wrong.

        The original concern does not survive contact with the index:

          * the UNIQUE index is scoped to ``dedup_gate='enforce'``;
          * legacy rows have ``dedup_gate IS NULL``, so they are OUTSIDE
            the index's scope entirely;
          * therefore two legacy rows sharing a canonical hash CANNOT
            violate the index.

        Verified on a copy of the live DB (2026-10-31): backfilling all
        12 345 recomputable rows took 1.5 s, changed ZERO original
        columns of ZERO existing rows, and the UNIQUE index still
        created successfully.

        WHAT "RECOMPUTABLE" MEANS
        -------------------------
        The row's ``payload_json`` must carry all three of ``content``,
        ``group_id`` and ``source_description``. The canonical hash is
        computed FROM those three, so a row missing one of them cannot
        be hashed and is left alone. Measured on the live DB: all 12 713
        rows have a complete triple, so nothing is skipped in practice.

        The legacy ``payload_hash`` column is PRESERVED verbatim — it is
        never overwritten. ``canonical_hash`` is a separate column
        precisely so the unknown-scheme digests stay visible and
        untouched.
        """
        rows = conn.execute(
            "SELECT id, payload_json FROM episodes WHERE canonical_hash IS NULL"
        ).fetchall()
        if not rows:
            return
        updates = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            content = payload.get("content")
            group_id = payload.get("group_id")
            source_description = payload.get("source_description")
            if content is None or group_id is None or source_description is None:
                continue
            updates.append(
                (canonical_payload_hash(content, group_id, source_description), row["id"])
            )
        if not updates:
            return
        conn.executemany(
            "UPDATE episodes SET canonical_hash = ? WHERE id = ?",
            updates,
        )
        logger.info(
            "canonical_hash backfill: filled %d row(s) from their own payload "
            "triple (the legacy payload_hash column is left untouched)",
            len(updates),
        )

    @staticmethod
    def _ensure_unique_active_hash_index_sync(conn: sqlite3.Connection) -> None:
        """Create the UNIQUE partial index that makes the dedup gate atomic.

        The index is scoped to the ACTIVE statuses, so a row that has
        already been ingested never blocks a later write of the same
        content. Two writers in two processes can no longer both insert
        the same active payload: the second INSERT fails with
        ``sqlite3.IntegrityError`` instead of winning a race.

        If the index cannot be created (a pre-existing duplicate pair in
        the ACTIVE statuses — measured 0 on the live DB as of 2026-10-31),
        the failure is LOGGED, not raised. A queue that cannot open its
        database is worse than a queue without the extra guarantee: the
        Python-level check still runs, so the gate degrades to
        best-effort rather than breaking every write.
        """
        try:
            conn.execute(UNIQUE_INDEX_DDL)
        except sqlite3.IntegrityError as err:
            logger.warning(
                "could not create %s (pre-existing active duplicates?): %s — "
                "the dedup gate falls back to the Python-level check",
                UNIQUE_INDEX_NAME,
                err,
            )
        except sqlite3.Error as err:  # pragma: no cover - defensive
            logger.warning(
                "could not create %s: %s — the dedup gate falls back to "
                "the Python-level check",
                UNIQUE_INDEX_NAME,
                err,
            )

    async def enqueue(
        self,
        content: str,
        group_id: str = "graphiti",
        name: Optional[str] = None,
        source_description: str = "agent_interaction",
        episode_id: Optional[str] = None,
        dedup_mode: Optional[str] = None,
    ) -> str:
        """Enqueue a new episode payload into SQLite.

        ``dedup_mode`` overrides ``cfg.pipeline.dedup.mode`` for this call.
        ``None`` means "use the configured mode". The three modes:

          off      — no hash computed, no hash stored, nothing rejected.
                     Byte-for-byte the pre-1b behaviour.
          observe  — hash computed + stored, duplicates detected and
                     counted, but the episode is STILL enqueued.
          enforce  — duplicates are rejected (raises
                     ``DuplicateEpisodeError``).

        When a hash IS stored, ``claimed_by`` is stamped as
        ``"mcp:<pid>"`` so the row's owner is visible in the same column the
        daemon writer uses. This is bookkeeping only — nothing reads
        ``claimed_by`` as a filter (see ``claim_next_batch``, which selects
        purely on ``status='queued'``).
        """
        mode = _resolve_dedup_mode(dedup_mode)
        ep_id = episode_id or str(uuid.uuid4())
        payload = {
            "content": content,
            "group_id": group_id,
            "name": name or f"ep_{ep_id[:8]}",
            "source_description": source_description,
        }
        payload_json = json.dumps(payload)
        now = _iso_now()

        # The CANONICAL hash is always computed: it is a content
        # address, not a gate. The mode decides whether a duplicate is
        # REJECTED, not whether the content can be addressed. Rows
        # written in ``off`` mode still carry a canonical hash so the
        # processing-layer backstop can see them — otherwise the whole
        # pre-migration population would be invisible to it.
        canonical_hash = canonical_payload_hash(content, group_id, source_description)

        # Hashing the LEGACY column is decided by MODE, independently of
        # rejecting. An observe-mode run must store it (otherwise the
        # next run cannot detect anything) while still enqueueing the
        # row. ``off`` keeps the pre-1b contract: no legacy hash.
        payload_hash: Optional[str] = None
        if mode != "off":
            payload_hash = canonical_hash

        if payload_hash is None:
            # off: no legacy hash column written, but canonical_hash is.
            async with self._write_lock:
                await asyncio.to_thread(
                    self._insert_sync,
                    ep_id=ep_id,
                    payload_json=payload_json,
                    now=now,
                    payload_hash=None,
                    dedup_gate="off",
                    canonical_hash=canonical_hash,
                )
            return ep_id

        # ── hashed path (observe / enforce) ───────────────────────────
        #
        # TWO layers now, because one layer cannot cover both writers:
        #
        #   1. the Python pre-check (fast, gives a clean "existing id"),
        #   2. the UNIQUE partial index (authoritative, race-proof).
        #
        # The pre-check alone was NOT enough: it is a SELECT followed by an
        # INSERT, and there are THREE writer processes (MCP stdio server,
        # the hook-spawned python, the daemon). The in-process asyncio lock
        # protects nothing across processes, so two writers could both pass
        # the check and both insert. Measured 2026-10-31: 5 such
        # cross-writer duplicate groups exist in the live history.
        #
        # The index alone was not enough either: it cannot tell the caller
        # WHICH row already holds the payload, and the enqueue-layer
        # contract is "return the existing id so the writer can say
        # 'already queued'". So the pre-check answers the question and the
        # index guarantees the answer stays true.
        async with self._write_lock:
            duplicate_of = await asyncio.to_thread(
                self._find_active_duplicate_sync, payload_hash
            )
            if duplicate_of is not None and mode == "enforce":
                # Rejected: do NOT insert. The caller gets the existing
                # episode id so it can report "already queued".
                await asyncio.to_thread(
                    self._record_dedup_event_sync,
                    mode,
                    "rejected",
                    payload_hash,
                    duplicate_of,
                    ep_id,
                )
                raise DuplicateEpisodeError(
                    existing_episode_id=duplicate_of, payload_hash=payload_hash
                )

            try:
                await asyncio.to_thread(
                    self._insert_sync,
                    ep_id=ep_id,
                    payload_json=payload_json,
                    now=now,
                    payload_hash=payload_hash,
                    dedup_gate=mode,
                    canonical_hash=canonical_hash,
                )
            except DuplicateRaceError as err:
                # The UNIQUE index rejected the INSERT: another writer
                # inserted the same active payload between our pre-check
                # and our INSERT. This is the race the index exists to
                # close. The row we wanted is NOT in the table.
                if mode == "enforce":
                    await asyncio.to_thread(
                        self._record_dedup_event_sync,
                        mode,
                        "rejected",
                        payload_hash,
                        err.existing_episode_id,
                        ep_id,
                    )
                    raise DuplicateEpisodeError(
                        existing_episode_id=err.existing_episode_id,
                        payload_hash=payload_hash,
                    ) from err
                # observe mode: the duplicate IS enqueued (that is the
                # whole point of observe), and the event is recorded.
                await asyncio.to_thread(
                    self._record_dedup_event_sync,
                    mode,
                    "detected",
                    payload_hash,
                    err.existing_episode_id,
                    ep_id,
                )
                return ep_id

        if duplicate_of is not None:
            # observe mode: logged + counted, but the row IS enqueued.
            await asyncio.to_thread(
                self._record_dedup_event_sync,
                mode,
                "detected",
                payload_hash,
                duplicate_of,
                ep_id,
            )
        return ep_id

    def _record_dedup_event_sync(
        self,
        mode: str,
        action: str,
        payload_hash: Optional[str],
        existing_episode_id: Optional[str],
        duplicate_episode_id: Optional[str],
    ) -> None:
        """Append one row to the durable ``dedup_events`` ledger.

        The counter used to be a plain Python attribute on the queue
        object (``self._dedup_hits``). The PicoClaw hook spawns a FRESH
        python process for every write, so the counter was always 0 when
        anybody read it — ``observed_hits`` was a number that LIES. The
        ledger row survives the process that produced it, so the observe
        rollout can finally be judged from a real number.

        ``self._dedup_hits`` is still incremented, but only as a
        process-local FALLBACK for the case where the ledger table is
        unavailable (a read-only DB, a migration that failed). The DB is
        the source of truth whenever it can be read.

        Never raises: a ledger write must not be able to fail an
        enqueue.
        """
        try:
            self._dedup_hits += 1
            with self._get_connection() as conn:
                record_dedup_event(
                    conn,
                    mode=mode,
                    action=action,
                    payload_hash=payload_hash,
                    existing_episode_id=existing_episode_id,
                    duplicate_episode_id=duplicate_episode_id,
                    writer=f"mcp:{os.getpid()}",
                )
            logger.info(
                "enqueue dedup %s/%s: duplicate of episode %s (hash=%s) — %s",
                mode,
                action,
                existing_episode_id,
                (payload_hash or "")[:12],
                "rejected" if action == "rejected" else "enqueued anyway",
            )
        except Exception:  # pragma: no cover - defensive only
            logger.debug("could not record dedup event", exc_info=True)

    def _find_active_duplicate_sync(self, payload_hash: str) -> Optional[str]:
        """Return episode_id of a row with this hash in an ACTIVE status.

        Active = ``queued`` or ``processing``. Rows in ``done`` / ``failed``
        / ``aborted`` do NOT block: re-queueing after a permanent failure
        is a legitimate retry, and blocking it would strand the row.

        The lookup itself lives in ``core.dedup`` so the MCP writer and
        the daemon writer search the SAME population. They used to carry
        separate copies of this query, and separate copies of the
        "which statuses are active" answer — which is precisely how the
        two writers drifted apart in the first place.
        """
        with self._get_connection() as conn:
            return find_active_duplicate(conn, payload_hash)

    def _insert_sync(
        self,
        ep_id: str,
        payload_json: str,
        now: str,
        payload_hash: Optional[str],
        dedup_gate: str = "off",
        canonical_hash: Optional[str] = None,
    ) -> None:
        """INSERT one episode row, optionally stamping the payload hash.

        ``payload_hash`` and ``claimed_by`` are written in the SAME
        statement as the row itself. Splitting them would leave a window
        where the row is visible to another writer WITHOUT its hash — and
        that window is exactly what breaks dedup.

        TWO HASH COLUMNS, TWO JOBS
        -------------------------
        ``payload_hash`` (LEGACY) is stamped only when the mode is not
        ``off``. Its "off means no hash at all" behaviour is the
        pre-1b contract and the 1b test suite pins it.

        ``canonical_hash`` (AUTHORITATIVE) is stamped ALWAYS, in every
        mode, because it is NOT a gate — it is a content address. The
        mode decides whether a duplicate is REJECTED; the hash only says
        what the content is.

        That distinction matters concretely: the processing-layer
        backstop asks "was this exact content already ingested?". If a
        row written in ``off`` mode carried no hash, the backstop could
        not see it and would answer "no" for the entire pre-migration
        population — a silent hole. Hashing every row costs one sha256
        per write and closes it.

        The UNIQUE index is unaffected: it only covers rows whose
        ``dedup_gate = 'enforce'``, so ``off``-mode rows stay
        duplicate-able and observe-mode rows stay duplicate-able (observe
        must never reject).

        Raises ``DuplicateRaceError`` when the UNIQUE partial index
        rejects the INSERT — i.e. another writer inserted the same
        ACTIVE payload between the pre-check and this statement. The
        enqueue path translates it into the mode's policy.
        """
        with self._get_connection() as conn:
            if payload_hash is None:
                # off mode: no legacy hash, but the canonical content
                # address is still stamped (see the docstring).
                conn.execute(
                    """
                    INSERT INTO episodes (
                        id, status, payload_json, enqueued_at, attempt_count,
                        consecutive_failures, created_at, updated_at,
                        dedup_gate, canonical_hash
                    ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?, ?, ?)
                    """,
                    (ep_id, payload_json, now, now, now, dedup_gate, canonical_hash),
                )
            else:
                try:
                    conn.execute(
                        """
                        INSERT INTO episodes (
                            id, status, payload_json, enqueued_at, attempt_count,
                            consecutive_failures, created_at, updated_at,
                            payload_hash, claimed_by, canonical_hash, dedup_gate
                        ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ep_id,
                            payload_json,
                            now,
                            now,
                            now,
                            payload_hash,
                            f"mcp:{os.getpid()}",
                            canonical_hash if canonical_hash is not None else payload_hash,
                            dedup_gate,
                        ),
                    )
                except sqlite3.IntegrityError as err:
                    # The UNIQUE index fired. Find out WHO holds the hash so
                    # the caller can be told the existing episode id.
                    #
                    # Two lookups: the ACTIVE one first (the usual case),
                    # then a status-agnostic fallback for the rare case
                    # where the holder left the active set in the
                    # microseconds between the rejected INSERT and this
                    # lookup. The fallback names a REAL row that really
                    # holds this payload, so the reported id is truthful
                    # even in that race.
                    existing = find_active_duplicate(conn, payload_hash)
                    if existing is None:
                        existing = find_any_duplicate(conn, payload_hash)
                    raise DuplicateRaceError(existing, payload_hash) from err
            conn.commit()

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
                        #
                        # The UPDATE moves the row INTO the UNIQUE index's
                        # scope, so it can conflict (the canonical hash is
                        # already held by another active row). A SAVEPOINT
                        # keeps that conflict local to THIS row: without it
                        # the exception would abort the whole reap batch.
                        conn.execute("SAVEPOINT janus_dedup_reap_stuck")
                        try:
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
                        except sqlite3.IntegrityError as err:
                            conn.execute("ROLLBACK TO janus_dedup_reap_stuck")
                            conn.execute("RELEASE janus_dedup_reap_stuck")
                            logger.info(
                                "reap skipped episode %s: canonical hash already "
                                "held by an active row (%s)",
                                ep_id,
                                err,
                            )
                            # Not a failure — the payload is in flight
                            # elsewhere. Mark it terminal so the reaper
                            # does not pick it up again every tick, but do
                            # NOT send it to the DLQ.
                            conn.execute(
                                """
                                UPDATE episodes
                                SET status = ?, updated_at = ?, last_error = ?
                                WHERE id = ?
                                """,
                                (
                                    BACKSTOP_SKIPPED_STATUS,
                                    now,
                                    "reaped: canonical hash already held by an active row",
                                    ep_id,
                                ),
                            )
                            reaped_count += 1
                            continue
                        conn.execute("RELEASE janus_dedup_reap_stuck")
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
                # (2) refresh ``enqueued_at`` — the clause that closes the
                # self-feeding loop. ``attempt_count`` is NOT reset (it is
                # the escalation signal read by ``should_retry``);
                # ``consecutive_failures`` IS reset, because it is written in
                # exactly one place and read in zero, so resetting it makes
                # the column mean what its name says.
                #
                # The UPDATE moves rows INTO the UNIQUE index's scope, so a
                # row whose canonical hash is already held by an active row
                # would raise IntegrityError. ``requeue_guarded`` runs each
                # row in a SAVEPOINT so one conflict cannot abort the whole
                # batch — without that, a single conflicting row would
                # leave the other N-1 rows unreaped for the night.
                requeued = requeue_guarded(
                    conn,
                    ids,
                    now=now,
                    extra_set="consecutive_failures = 0",
                )
                conn.commit()
                return len(requeued)

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
                # The UPDATE moves the row INTO the UNIQUE index's scope,
                # so it can conflict with another active row holding the
                # same canonical hash. A SAVEPOINT keeps the conflict
                # local: without it the IntegrityError would abort the
                # replay AND leave the dead_letter row marked recovered
                # while the episode is still dead.
                requeued = requeue_guarded(
                    conn,
                    [episode_id],
                    now=now,
                    extra_set="attempt_count = 0, last_error = NULL, last_replay_at = ?",
                    extra_params=[now],
                )
                if not requeued:
                    conn.rollback()
                    return False
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
                    #
                    # The UPDATE moves the row INTO the UNIQUE index's
                    # scope, so it can conflict with another active row
                    # holding the same canonical hash. A SAVEPOINT keeps
                    # the conflict local to this row — otherwise one
                    # conflicting row would abort the WHOLE batch replay.
                    conn.execute("SAVEPOINT janus_dedup_replay_batch")
                    try:
                        cur = conn.execute(
                            "UPDATE episodes SET status = 'queued', attempt_count = 0,"
                            " last_error = NULL, enqueued_at = ?, updated_at = ?, last_replay_at = ?"
                            " WHERE id = ? AND status != 'done'",
                            (now, now, now, ep_id),
                        )
                    except sqlite3.IntegrityError as err:
                        conn.execute("ROLLBACK TO janus_dedup_replay_batch")
                        conn.execute("RELEASE janus_dedup_replay_batch")
                        logger.info(
                            "dlq batch replay skipped episode %s: canonical hash "
                            "already held by an active row (%s)",
                            ep_id,
                            err,
                        )
                        # The row stays dead — it was NOT replayed, so the
                        # dead_letter row must stay open (not marked
                        # recovered). Leaving it open is correct: the
                        # episode is still not done.
                        continue
                    conn.execute("RELEASE janus_dedup_replay_batch")
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

    def get_dedup_stats(self) -> Dict[str, Any]:
        """Return dedup counters for the observe-mode rollout.

        Deliberately NOT merged into :meth:`get_stats` — that dict is summed
        by callers (``dream.py`` computes ``done + queued``) and adding a
        non-status key would corrupt those sums. The dedup numbers live in
        their own accessor so nobody has to remember which dict is which.

        Keys:
          mode              — the effective mode for the NEXT enqueue.
          observed_hits     — duplicates DETECTED, read from the durable
                              ``dedup_events`` ledger. It is a CROSS-PROCESS
                              number: it survives the process that detected
                              the duplicate, which is the whole point — the
                              PicoClaw hook spawns a fresh python per write,
                              so a RAM counter was always 0 here.
          hashed_rows       — rows in the DB carrying a payload_hash.
          canonical_rows    — rows carrying a canonical_hash (the
                              enforceable column).
          active_duplicates — rows sharing a payload_hash while still
                              active (queued/processing). This is the number
                              that says "the enforce gate would reject N
                              rows right now".
          backstop_skips    — rows skipped by the processing-layer
                              backstop (i.e. content that had already been
                              ingested when the row was claimed).

        ``observed_hits`` counts EVENTS, not distinct payloads: the same
        payload detected twice is two events. That matches what an
        operator wants to know ("how much duplicate traffic did we see"),
        not "how many distinct payloads were duplicated".
        """
        with self._get_connection() as conn:
            hashed = conn.execute(
                "SELECT COUNT(*) as count FROM episodes WHERE payload_hash IS NOT NULL"
            ).fetchone()["count"]
            canonical = conn.execute(
                "SELECT COUNT(*) as count FROM episodes WHERE canonical_hash IS NOT NULL"
            ).fetchone()["count"]
            active_dupes = conn.execute(
                "SELECT COUNT(*) as count FROM ("
                "  SELECT payload_hash FROM episodes"
                "  WHERE payload_hash IS NOT NULL"
                "    AND status IN ('queued', 'processing')"
                "  GROUP BY payload_hash HAVING COUNT(*) > 1"
                ")"
            ).fetchone()["count"]
            # Durable counter. Read from the ledger, NOT from the RAM
            # attribute: the ledger survives the process that wrote it.
            try:
                ledger_hits = conn.execute(
                    "SELECT COUNT(*) as count FROM dedup_events"
                ).fetchone()["count"]
            except sqlite3.Error:
                # The ledger table is missing (a DB created before this
                # migration and never re-opened). Fall back to the
                # process-local counter rather than failing the read.
                ledger_hits = None
            try:
                backstop_skips = conn.execute(
                    "SELECT COUNT(*) as count FROM episodes WHERE status = ?",
                    (BACKSTOP_SKIPPED_STATUS,),
                ).fetchone()["count"]
            except sqlite3.Error:
                backstop_skips = 0

        observed_hits = ledger_hits if ledger_hits is not None else self._dedup_hits
        return {
            "mode": _resolve_dedup_mode(),
            "observed_hits": int(observed_hits),
            "hashed_rows": int(hashed),
            "canonical_rows": int(canonical),
            "active_duplicates": int(active_dupes),
            "backstop_skips": int(backstop_skips),
        }

    def get_recent_dedup_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent dedup ledger rows, newest first.

        The counter says HOW MANY; this says WHAT. An operator deciding
        whether to switch to ``enforce`` needs to see the payloads and
        the writer that produced them, not just a number.
        """
        try:
            with self._get_connection() as conn:
                return recent_dedup_events(conn, limit=limit)
        except sqlite3.Error:
            return []

    def find_ingested_duplicate_sync(self, payload_hash: str) -> Optional[str]:
        """Return the id of a row whose payload was ALREADY INGESTED.

        "Ingested" = ``status = 'done'`` — the content reached the graph.
        This is the lookup the processing-layer backstop uses, and it is
        deliberately WIDER than the enqueue gate's active-only lookup: by
        the time the backstop runs, the first row has usually left the
        active set, so an active-only check would find nothing.

        It is also the risky one: suppressing a legitimate re-ingestion
        loses information. That is why the backstop is opt-in and
        default-off (see ``config.DedupBackstopSettings``).
        """
        if not payload_hash:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM episodes WHERE "
                "(canonical_hash = ? OR (canonical_hash IS NULL AND payload_hash = ?)) "
                "AND status = 'done' "
                "ORDER BY created_at DESC LIMIT 1",
                (payload_hash, payload_hash),
            ).fetchone()
            if row is None:
                return None
            return row[0] if not isinstance(row, sqlite3.Row) else row["id"]

    async def mark_skipped_duplicate(
        self,
        episode_id: str,
        existing_episode_id: Optional[str] = None,
        payload_hash: Optional[str] = None,
    ) -> None:
        """Mark a claimed row as skipped because its payload was already ingested.

        The row gets its OWN status (``skipped_duplicate``), not ``done``:

          * not ``done`` — that status means "this content WAS ingested",
            and the backstop explicitly did NOT ingest it. Reusing
            ``done`` would inflate ``dream.py``'s ``done + queued``
            episode count with content the graph never saw;
          * not ``failed`` / ``aborted`` — those are failure statuses.
            The reapers pick them up and either requeue them or push
            them to the DLQ, so a benign skip would turn into queue
            pressure and DLQ noise.

        The row is TERMINAL: it is never claimed again (``claim_next_batch``
        selects ``status='queued'``) and it is never dead-lettered (the
        DLQ is written by ``mark_failed`` / ``mark_aborted`` only).
        """
        now = _iso_now()

        def _sync_skip():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE episodes
                    SET status = ?, finished_at = ?, updated_at = ?,
                        last_error = ?, consecutive_failures = 0
                    WHERE id = ?
                    """,
                    (
                        BACKSTOP_SKIPPED_STATUS,
                        now,
                        now,
                        (
                            "skipped: payload already ingested as "
                            + (existing_episode_id or "unknown")
                        ),
                        episode_id,
                    ),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_sync_skip)

        if existing_episode_id or payload_hash:
            try:
                with self._get_connection() as conn:
                    record_dedup_event(
                        conn,
                        mode="backstop",
                        action="backstop_skipped",
                        payload_hash=payload_hash,
                        existing_episode_id=existing_episode_id,
                        duplicate_episode_id=episode_id,
                        writer=f"worker:{os.getpid()}",
                    )
            except Exception:  # pragma: no cover - defensive only
                logger.debug("could not record backstop dedup event", exc_info=True)

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
