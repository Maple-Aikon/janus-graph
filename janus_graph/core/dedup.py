"""Shared dedup vocabulary for every episode writer.

WHY THIS MODULE EXISTS
-----------------------
Two processes write into the same ``episodes`` table:

  1. the MCP stdio writer (``pipeline/queue.py::EpisodeQueue.enqueue``);
  2. the daemon HTTP writer
     (``daemon/episode_queue_adapter.py::EpisodeQueueAdapter.enqueue_guarded``).

Before 2026-10-31 they each carried their OWN copy of the dedup
vocabulary — their own "which statuses count as active", their own
"how do I look up a duplicate", their own "how do I stamp a hash".
That is exactly the failure class this whole exercise exists to kill:
the two writers disagreed in the past (the MCP writer wrote no hash at
all, so the daemon's gate could never see MCP rows) and the only reason
they agree today is a shared *hash function* — a shared *policy* was
never factored out.

So the policy lives HERE, and both writers import it. If the definition
of "active" changes, it changes in one place.

THE THREE THINGS THAT MUST AGREE
---------------------------------
1. ``ACTIVE_STATUSES`` — which statuses mean "this payload is in
   flight". Used by the Python-level pre-check AND mirrored literally in
   the partial UNIQUE index predicate. If these two drift, the DB
   invariant and the Python check would disagree about the same row.

2. ``find_active_duplicate`` — the lookup. Both writers call it, so
   they see the same population.

3. ``canonical_payload_hash`` — the hash scheme itself (imported from
   ``core.payload_hash``, which owns it).

THE CANONICAL / LEGACY COLUMN SPLIT
------------------------------------
``episodes.payload_hash`` is the LEGACY column. It was written by the
daemon adapter and by the 1b MCP path, but the 11 482
``signet_*_reimport`` backfill rows carry a hash from an UNKNOWN scheme
(measured 2026-10-31 on the live queue DB: 367 rows match the canonical
scheme, 11 482 do not, and none of 6 plausible sha256 orderings
reproduces the legacy digests).

``episodes.canonical_hash`` is the AUTHORITATIVE column for new writes
AND for every row whose ``payload_json`` carries a complete
``(content, group_id, source_description)`` triple. It is RECOMPUTED
from that triple, so it does not depend on the legacy column's scheme
at all — which matters because the 11 482 ``signet_*_reimport``
backfill rows carry a hash from an UNKNOWN scheme (measured 2026-10-31
on the live queue DB: 367 rows match the canonical scheme, 11 482 do
not, and none of 6 plausible sha256 orderings reproduces the legacy
digests).

The legacy ``payload_hash`` column is PRESERVED verbatim. It is never
overwritten and never deleted: it is the historical record of what the
old writers stamped, and the backfill does not need it.

Because the canonical hash is recomputed from the payload triple, the
legacy rows are NOT invisible to the backstop — they carry a canonical
hash like every other row. That was the reason the FIRST cut of the
backfill (which only filled "provably canonical" rows) was too narrow:
it would have left the entire pre-migration population invisible to the
processing-layer backstop.

The UNIQUE index only covers rows written under the ``enforce`` gate,
so backfilling legacy rows (whose ``dedup_gate`` is NULL) cannot make
them collide with each other. Verified on a copy of the live DB
(2026-10-31): backfilling all 12 345 recomputable rows changed ZERO
original columns of ZERO existing rows, and the index still created
successfully.

The lookup therefore matches on BOTH: the canonical column first, and
the legacy column for rows that never got a canonical hash (e.g. a row
whose payload triple is incomplete). That keeps the daemon's gate able
to see legacy rows while the new column carries the enforceable
invariant.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from .payload_hash import canonical_payload_hash

logger = logging.getLogger("janus_graph.core.dedup")

__all__ = [
    "ACTIVE_STATUSES",
    "NON_BLOCKING_STATUSES",
    "BACKSTOP_SKIPPED_STATUS",
    "UNIQUE_INDEX_NAME",
    "UNIQUE_INDEX_DDL",
    "canonical_payload_hash",
    "find_active_duplicate",
    "find_any_duplicate",
    "record_dedup_event",
    "recent_dedup_events",
    "requeue_guarded",
]


#: Statuses that mean "this payload is currently in flight". A duplicate
#: of one of these is what the enqueue gate rejects in ``enforce`` mode.
#:
#: These three literals are ALSO spelled out in ``UNIQUE_INDEX_DDL``
#: because SQLite does not accept a bound parameter in a partial index
#: predicate. The two lists must be changed together — ``test_`` coverage
#: in tests/test_dedup_hardening.py asserts the index predicate and
#: ``ACTIVE_STATUSES`` name the same set.
ACTIVE_STATUSES: Sequence[str] = ("queued", "processing")

#: Statuses that do NOT block a new enqueue: re-queueing after a
#: permanent failure is a legitimate retry, and blocking it would strand
#: the row forever.
NON_BLOCKING_STATUSES: Sequence[str] = ("done", "failed", "aborted")

#: Status given to a row the processing-layer backstop decided to skip.
#:
#: It is deliberately NOT one of the failure statuses: the row was not a
#: failure, so it must not be reaped, retried or dead-lettered. It is
#: also NOT 'done' — 'done' means "this content WAS ingested", and
#: claiming it for a row that never was would inflate ``dream.py``'s
#: ``done + queued`` episode count with content the graph never saw.
BACKSTOP_SKIPPED_STATUS = "skipped_duplicate"

#: Name of the UNIQUE partial index that makes the dedup gate atomic.
#:
#: The predicate is deliberately NARROWER than "any active row with this
#: hash". It only covers rows written while the gate was in ``enforce``
#: mode (``dedup_gate = 'enforce'``). Reason: ``observe`` mode MUST
#: still enqueue a duplicate — that is its entire contract ("count what
#: the gate would have caught, reject nothing"). A plain "any active row"
#: index would make observe mode reject, which is the exact opposite of
#: what observe means.
#:
#: So the invariant is: two rows written under ENFORCE can never share an
#: active canonical hash. That is the race that matters — the race is
#: between two writers that both believe they are the gate. Rows written
#: under observe are not gate-protected, by design.
#:
#: The predicate also scopes to the ACTIVE statuses, so a row that has
#: already been ingested (status='done') does not block a later write of
#: the same content — that is the "done-aware" freedom the enqueue gate
#: has always had.
UNIQUE_INDEX_NAME = "idx_episodes_canonical_hash_active"

#: DDL for that index. The predicate must list ACTIVE_STATUSES literally —
#: SQLite does not accept a bind parameter in a partial index predicate.
UNIQUE_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_canonical_hash_active "
    "ON episodes(canonical_hash) "
    "WHERE canonical_hash IS NOT NULL "
    "AND dedup_gate = 'enforce' "
    "AND status IN ('queued', 'processing')"
)

#: Values of the ``episodes.dedup_gate`` column — which gate wrote the
#: row. This is bookkeeping that also decides whether the UNIQUE index
#: covers the row.
#:
#: ``off``     — written with no hash at all (pre-1b behaviour).
#: ``observe`` — hash stored, duplicate detected and counted, duplicate
#:               STILL enqueued. Not gate-protected.
#: ``enforce`` — hash stored, duplicate REJECTED. Gate-protected by the
#:               UNIQUE index.
DEDUP_GATE_OFF = "off"
DEDUP_GATE_OBSERVE = "observe"
DEDUP_GATE_ENFORCE = "enforce"


def _active_placeholders() -> str:
    """Render ``ACTIVE_STATUSES`` as a SQL literal list.

    The values are module constants (never user input), so inlining them
    is safe; a bound parameter cannot be used because SQLite rejects
    parameters in an ``IN`` list that is built at query-preparation time
    in some paths. Rendering them keeps the SQL text and
    ``ACTIVE_STATUSES`` provably in sync.
    """
    return ", ".join(f"'{s}'" for s in ACTIVE_STATUSES)


#: SQL fragment matching a row whose canonical (or legacy) hash equals
#: the given digest AND whose status is active.
_DUP_MATCH_SQL = (
    "(canonical_hash = ? OR (canonical_hash IS NULL AND payload_hash = ?)) "
    f"AND status IN ({_active_placeholders()})"
)


def find_active_duplicate(
    conn: sqlite3.Connection,
    payload_hash: str,
) -> Optional[str]:
    """Return the id of an ACTIVE row holding this payload, else ``None``.

    "Active" = ``status IN ACTIVE_STATUSES`` (queued / processing). Rows
    that already reached ``done`` / ``failed`` / ``aborted`` do NOT
    block: re-queueing after a permanent failure is a legitimate retry,
    and blocking it would strand the row forever.

    The lookup matches the canonical column OR the legacy column, but
    the legacy column only for rows that never received a canonical
    hash. That keeps legacy rows visible to the gate without letting a
    legacy digest shadow a canonical one.

    Ordering: newest first (``created_at DESC``), matching the
    pre-2026-10-31 behaviour, so the reported "existing id" is the most
    recent holder of the payload.
    """
    if not payload_hash:
        return None
    row = conn.execute(
        "SELECT id FROM episodes WHERE "
        + _DUP_MATCH_SQL
        + " ORDER BY created_at DESC LIMIT 1",
        (payload_hash, payload_hash),
    ).fetchone()
    if row is None:
        return None
    return row[0] if not isinstance(row, sqlite3.Row) else row["id"]


def find_any_duplicate(
    conn: sqlite3.Connection,
    payload_hash: str,
) -> Optional[str]:
    """Return the id of ANY row holding this hash, whatever its status.

    This is the FALLBACK for the rare case where the UNIQUE index fires
    but the active-status lookup comes up empty — which happens when the
    holder left the active set (was claimed, finished or failed) in the
    microseconds between the rejected INSERT and the lookup.

    The row existed a moment ago; naming it is more useful than saying
    "unknown", and the id is still truthful (it is a real row that
    really holds this payload). It is only used for REPORTING, never for
    deciding whether to reject — that decision is the index's.
    """
    if not payload_hash:
        return None
    row = conn.execute(
        "SELECT id FROM episodes WHERE "
        "(canonical_hash = ? OR (canonical_hash IS NULL AND payload_hash = ?)) "
        "ORDER BY created_at DESC LIMIT 1",
        (payload_hash, payload_hash),
    ).fetchone()
    if row is None:
        return None
    return row[0] if not isinstance(row, sqlite3.Row) else row["id"]


def record_dedup_event(
    conn: sqlite3.Connection,
    *,
    mode: str,
    action: str,
    payload_hash: Optional[str] = None,
    existing_episode_id: Optional[str] = None,
    duplicate_episode_id: Optional[str] = None,
    writer: Optional[str] = None,
    detected_at: Optional[str] = None,
) -> None:
    """Append one row to the durable ``dedup_events`` ledger.

    This is what makes the observe rollout measurable. The counter used
    to be a plain Python attribute on the queue object, so the PicoClaw
    hook — which spawns a FRESH python process for every write — reset
    it before anybody could read it. ``observed_hits`` was a number that
    LIES. Writing the event to the DB makes the count survive the
    process that produced it.

    ``action`` is one of:
      ``detected``       — a duplicate was found; the episode was still
                           enqueued (observe mode).
      ``rejected``       — a duplicate was found; the episode was NOT
                           enqueued (enforce mode).
      ``backstop_skipped`` — the processing-layer backstop decided the
                           payload had already been ingested.

    The function NEVER raises: a ledger write must not be able to fail
    an enqueue. Failures are logged at WARNING.
    """
    try:
        from datetime import datetime, timezone

        ts = detected_at or datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO dedup_events (
                detected_at, mode, action, payload_hash,
                existing_episode_id, duplicate_episode_id, writer
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                mode,
                action,
                payload_hash,
                existing_episode_id,
                duplicate_episode_id,
                writer,
            ),
        )
        conn.commit()
    except Exception as err:  # pragma: no cover - defensive only
        logger.warning("could not record dedup event: %s", err)


def recent_dedup_events(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list:
    """Return the most recent dedup ledger rows, newest first.

    Used by the MCP tool so an operator can see WHAT was detected, not
    just how many times.
    """
    rows = conn.execute(
        "SELECT detected_at, mode, action, payload_hash, existing_episode_id, "
        "duplicate_episode_id, writer FROM dedup_events "
        "ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    out = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            out.append({k: row[k] for k in row.keys()})
        else:  # pragma: no cover - plain tuple row
            out.append(
                {
                    "detected_at": row[0],
                    "mode": row[1],
                    "action": row[2],
                    "payload_hash": row[3],
                    "existing_episode_id": row[4],
                    "duplicate_episode_id": row[5],
                    "writer": row[6],
                }
            )
    return out


def requeue_guarded(
    conn: sqlite3.Connection,
    episode_ids: List[str],
    *,
    now: str,
    extra_set: str = "",
    extra_params: Optional[List[Any]] = None,
) -> List[str]:
    """Move rows back to ``status='queued'`` without violating the UNIQUE index.

    WHY THIS EXISTS
    ---------------
    Four paths write ``status='queued'``: the reaper for stuck rows, the
    reaper for failed/aborted rows, the single DLQ replay and the batch
    DLQ replay. All four are UPDATEs, not INSERTs — and an UPDATE that
    moves a row INTO the index's scope (``canonical_hash IS NOT NULL AND
    dedup_gate='enforce' AND status IN ('queued','processing')``) is
    rejected by SQLite exactly like an INSERT would be.

    Measured on SQLite (2026-10-31): ``UPDATE episodes SET status='queued'``
    on a row whose canonical hash is already held by an active row raises
    ``sqlite3.IntegrityError: UNIQUE constraint failed``. Left unguarded,
    that exception would escape into the sweep and abort the WHOLE batch
    of requeues — turning one conflicting row into a broken nightly reap.

    So each requeue runs inside a SAVEPOINT. On conflict the row is
    rolled back to its previous state, recorded as skipped, and the
    remaining rows still go through.

    A skipped row is NOT marked failed and NOT sent to the DLQ. It is
    not a failure: the payload is already in flight elsewhere, which is
    the whole point of the gate. Marking it failed would send it to the
    DLQ and burn retry budget on a benign condition.

    Returns the list of episode ids that were actually requeued.
    """
    requeued: List[str] = []
    for episode_id in episode_ids:
        conn.execute("SAVEPOINT janus_dedup_requeue")
        try:
            params: List[Any] = [now]
            sql = (
                "UPDATE episodes SET status = 'queued', enqueued_at = ?, updated_at = ?"
            )
            params.append(now)
            if extra_set:
                sql += ", " + extra_set
                params.extend(extra_params or [])
            sql += " WHERE id = ?"
            params.append(episode_id)
            conn.execute(sql, params)
        except sqlite3.IntegrityError as err:
            # Conflict: this row's canonical hash is already held by an
            # active row. Undo THIS row only, keep going with the rest.
            conn.execute("ROLLBACK TO janus_dedup_requeue")
            conn.execute("RELEASE janus_dedup_requeue")
            logger.info(
                "requeue skipped episode %s: its canonical hash is already "
                "held by an active row (%s)",
                episode_id,
                err,
            )
            continue
        conn.execute("RELEASE janus_dedup_requeue")
        requeued.append(episode_id)
    return requeued
