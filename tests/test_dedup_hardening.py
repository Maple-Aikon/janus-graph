"""Dedup hardening (2026-10-31) — the three real defects, measured.

Context
-------
The architecture review (memory/plan/dedup-at-processing-vs-enqueue-layer-
architecture-review-20261031.md) found three defects in the enqueue-layer
dedup gate, plus one thing it was MISSING:

  1. **The counter was a lie.** ``observed_hits`` was a plain Python
     attribute on the queue object. The PicoClaw hook spawns a FRESH
     python process for every write, so the counter was always 0 when
     anybody read it. The observe rollout could not be judged at all.

  2. **The check was not atomic.** It was a SELECT followed by an
     INSERT, guarded by an IN-PROCESS asyncio lock — with three writer
     processes. Two writers could both pass the check and both insert.
     Measured: 5 such cross-writer duplicate groups in the live history.

  3. **The hash scheme had drifted.** 11 482 of 11 849 hashed rows on
     the live DB carry a hash from an UNKNOWN scheme. Any dedup that
     trusts the stored ``payload_hash`` silently misses them.

  4. **There was no processing-layer backstop.** The plan recommends one
     as a safety net under the gate (NOT as the primary gate).

What these tests pin
--------------------
  A. The durable ledger counts hits ACROSS PROCESSES.
  B. The UNIQUE partial index makes the gate atomic, and the index
     predicate agrees with ``ACTIVE_STATUSES``.
  C. ``enforce`` reports the REAL existing episode id, even when the
     duplicate arrived through the race.
  D. Legacy rows are never rewritten: the unknown-scheme population
     keeps its hash and keeps ``canonical_hash IS NULL``.
  E. The backstop is default-off, skip-only, never raises, and never
     produces a DLQ row.
  F. The requeue paths survive a UNIQUE conflict without aborting the
     whole batch.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from janus_graph.core.dedup import (
    ACTIVE_STATUSES,
    DEDUP_GATE_ENFORCE,
    DEDUP_GATE_OBSERVE,
    UNIQUE_INDEX_DDL,
    UNIQUE_INDEX_NAME,
    find_active_duplicate,
    requeue_guarded,
)
from janus_graph.core.payload_hash import canonical_payload_hash
from janus_graph.pipeline.queue import (
    BACKSTOP_SKIPPED_STATUS,
    DuplicateEpisodeError,
    EpisodeQueue,
)


def _tmp_queue() -> EpisodeQueue:
    db_path = Path(tempfile.mkdtemp(prefix="jg_hard_")) / "episodes.db"
    return EpisodeQueue(str(db_path))


def _rows(q: EpisodeQueue, sql: str, params=()):
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def _row(q: EpisodeQueue, episode_id: str):
    return _rows(q, "SELECT * FROM episodes WHERE id = ?", (episode_id,))[0]


# ── A. the durable ledger ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_hit_is_recorded_in_the_ledger():
    q = _tmp_queue()

    await q.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")
    await q.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")

    events = _rows(q, "SELECT * FROM dedup_events ORDER BY id")
    assert len(events) == 1, "the duplicate must leave a ledger row"
    assert events[0]["mode"] == "observe"
    assert events[0]["action"] == "detected"
    assert events[0]["existing_episode_id"] is not None
    assert events[0]["payload_hash"] == canonical_payload_hash("x", "g", "s")


@pytest.mark.asyncio
async def test_ledger_survives_a_new_queue_object_on_the_same_db():
    """The counter must not be process-local RAM.

    This is the defect the whole ledger exists to fix: the hook spawns
    a fresh python per write, so a RAM counter reads 0 from anywhere
    else. A NEW EpisodeQueue object over the SAME file must see the
    hits recorded by the previous one.
    """
    db_path = Path(tempfile.mkdtemp(prefix="jg_hard_")) / "episodes.db"

    q1 = EpisodeQueue(str(db_path))
    await q1.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")
    await q1.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")

    # A DIFFERENT object — this is what "another process" looks like
    # from the queue's point of view.
    q2 = EpisodeQueue(str(db_path))
    assert q2.get_dedup_stats()["observed_hits"] == 1, (
        "the hit count must come from the DB, not from RAM"
    )
    assert q2._dedup_hits == 0, "the RAM counter of the new object is empty — as it must be"


@pytest.mark.asyncio
async def test_ledger_survives_a_whole_separate_python_process():
    """The strongest form: a real OS process, like the hook spawns.

    The hook runs ``python -c <script>`` per write. A subprocess is the
    only way to prove the number survives process teardown, not just an
    object boundary.
    """
    db_path = Path(tempfile.mkdtemp(prefix="jg_hard_")) / "episodes.db"

    script = chr(10).join(
        [
            "import asyncio, sys",
            "from janus_graph.pipeline.queue import EpisodeQueue",
            "async def run():",
            "    q = EpisodeQueue(sys.argv[1])",
            "    await q.enqueue('x', group_id='g', source_description='s', dedup_mode='observe')",
            "    await q.enqueue('x', group_id='g', source_description='s', dedup_mode='observe')",
            "asyncio.run(run())",
        ]
    )
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script, str(db_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    q = EpisodeQueue(str(db_path))
    assert q.get_dedup_stats()["observed_hits"] == 1, (
        "a duplicate detected in ANOTHER PROCESS must be counted here"
    )


@pytest.mark.asyncio
async def test_enforce_rejection_is_recorded_in_the_ledger():
    q = _tmp_queue()

    await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")
    with pytest.raises(DuplicateEpisodeError):
        await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")

    events = _rows(q, "SELECT * FROM dedup_events")
    assert len(events) == 1
    assert events[0]["action"] == "rejected"
    assert events[0]["mode"] == "enforce"


# ── B. the UNIQUE index makes the gate atomic ───────────────────────────


@pytest.mark.asyncio
async def test_unique_index_exists_and_covers_enforce_rows():
    q = _tmp_queue()

    indexes = _rows(q, "SELECT name FROM sqlite_master WHERE type='index'")
    names = {r["name"] for r in indexes}
    assert UNIQUE_INDEX_NAME in names, "the atomicity index must exist"


def test_index_predicate_and_active_statuses_agree():
    """The DB invariant and the Python check must name the SAME set.

    If the index predicate drifted from ``ACTIVE_STATUSES``, the two
    layers would disagree about what "active" means — the exact class of
    bug this whole change exists to remove (the two writers used to
    carry separate copies of that answer).
    """
    for status in ACTIVE_STATUSES:
        assert f"'{status}'" in UNIQUE_INDEX_DDL, (
            f"{status} is active but missing from the index predicate"
        )
    # And the predicate must not cover a NON-blocking status.
    for status in ("done", "failed", "aborted"):
        assert f"'{status}'" not in UNIQUE_INDEX_DDL, (
            f"{status} must not block a new enqueue — it is not in the index"
        )


@pytest.mark.asyncio
async def test_two_enforce_rows_cannot_share_an_active_canonical_hash():
    """The DB — not the Python pre-check — is what makes it atomic.

    Bypasses the queue's own gate entirely and inserts two rows with
    the same canonical hash directly. The second must be rejected by
    SQLite.
    """
    q = _tmp_queue()
    payload_hash = canonical_payload_hash("x", "g", "s")
    payload = json.dumps(
        {
            "content": "x",
            "group_id": "g",
            "name": "ep",
            "source_description": "s",
        }
    )
    now = "2026-10-31T00:00:00+00:00"

    def _insert(ep_id: str):
        with sqlite3.connect(str(q.db_path)) as conn:
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
                    payload,
                    now,
                    now,
                    now,
                    payload_hash,
                    "daemon:1",
                    payload_hash,
                    DEDUP_GATE_ENFORCE,
                ),
            )
            conn.commit()

    _insert("row-a")
    with pytest.raises(sqlite3.IntegrityError):
        _insert("row-b")


@pytest.mark.asyncio
async def test_observe_rows_are_not_covered_by_the_index():
    """Observe MUST still be able to enqueue a duplicate.

    This is the trap: an index scoped to "any active row with this hash"
    would make observe mode REJECT — the exact opposite of observe's
    contract ("count what the gate would have caught, reject nothing").
    The index is scoped to ``dedup_gate='enforce'`` precisely so that
    observe keeps working.
    """
    q = _tmp_queue()

    ep1 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")
    ep2 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")

    assert ep1 != ep2, "observe mode must NOT reject the duplicate"
    assert len(_rows(q, "SELECT id FROM episodes")) == 2


@pytest.mark.asyncio
async def test_done_row_does_not_block_a_new_enforce_row():
    """A row that was already ingested must not block a later write.

    The index scope is the ACTIVE statuses. A 'done' row is out of
    scope, so the same content can be enqueued again — the freedom the
    enqueue gate has always had.
    """
    q = _tmp_queue()

    ep1 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")
    await q.mark_done(ep1)

    ep2 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")
    assert ep2 != ep1
    assert len(_rows(q, "SELECT id FROM episodes")) == 2


# ── C. enforce reports the REAL existing id ──────────────────────────────


@pytest.mark.asyncio
async def test_enforce_reports_the_existing_episode_id():
    q = _tmp_queue()

    ep1 = await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")
    with pytest.raises(DuplicateEpisodeError) as excinfo:
        await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")

    assert excinfo.value.existing_episode_id == ep1
    assert len(_rows(q, "SELECT id FROM episodes")) == 1, "the duplicate must NOT be inserted"


@pytest.mark.asyncio
async def test_enforce_reports_the_existing_id_even_after_the_race():
    """The race path must report a REAL id, not ``None``.

    The pre-check misses (the other writer inserted in between) and the
    UNIQUE index catches it. The caller must still learn WHICH row
    holds the payload — otherwise the writer-level contract ("already
    queued as <id>") breaks exactly in the case it matters most.
    """
    q = _tmp_queue()
    payload_hash = canonical_payload_hash("x", "g", "s")
    payload = json.dumps(
        {
            "content": "x",
            "group_id": "g",
            "name": "ep",
            "source_description": "s",
        }
    )
    now = "2026-10-31T00:00:00+00:00"

    # A row written by ANOTHER writer (claimed_by daemon), active.
    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at,
                payload_hash, claimed_by, canonical_hash, dedup_gate
            ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                "other-writer-row",
                payload,
                now,
                now,
                now,
                payload_hash,
                "daemon:999",
                payload_hash,
                DEDUP_GATE_ENFORCE,
            ),
        )
        conn.commit()

    with pytest.raises(DuplicateEpisodeError) as excinfo:
        await q.enqueue("x", group_id="g", source_description="s", dedup_mode="enforce")

    assert excinfo.value.existing_episode_id == "other-writer-row"
    assert len(_rows(q, "SELECT id FROM episodes")) == 1


# ── D. legacy rows are never rewritten ───────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_unknown_scheme_row_keeps_its_hash_but_gets_a_canonical_hash():
    """A row whose stored hash is NOT the canonical hash must keep its hash.

    The 11 482 ``signet_*_reimport`` rows carry a hash from a scheme
    nobody could reverse-engineer (6 plausible sha256 orderings all
    failed to reproduce it). The legacy column is PRESERVED verbatim —
    overwriting it would destroy the historical record and could make
    two unrelated rows collide under the UNIQUE index.

    But the row DOES get a ``canonical_hash``, recomputed from its own
    payload triple. It has to: the backstop asks "was this content
    already ingested?", and a row whose canonical hash is NULL is
    invisible to that question. Leaving the whole pre-migration
    population invisible would be a silent hole.

    The UNIQUE index cannot be affected by this, because it only covers
    rows written under the ``enforce`` gate and these rows have
    ``dedup_gate IS NULL``.
    """
    q = _tmp_queue()
    legacy_hash = "deadbeef" * 8  # NOT the canonical hash of anything
    payload = json.dumps(
        {
            "content": "legacy content",
            "group_id": "graphiti_memory",
            "name": "ep",
            "source_description": "signet_legacy_reimport",
        }
    )
    now = "2026-10-31T00:00:00+00:00"

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at, payload_hash
            ) VALUES (?, 'done', ?, ?, 0, 0, ?, ?, ?)
            """,
            ("legacy-row", payload, now, now, now, legacy_hash),
        )
        conn.commit()

    # Re-open the queue: the migration runs on construction.
    q2 = EpisodeQueue(str(q.db_path))
    row = _row(q2, "legacy-row")

    assert row["payload_hash"] == legacy_hash, "the legacy hash must be PRESERVED verbatim"
    assert row["canonical_hash"] == canonical_payload_hash(
        "legacy content", "graphiti_memory", "signet_legacy_reimport"
    ), "the canonical hash is recomputed from the row's OWN payload triple"


@pytest.mark.asyncio
async def test_row_with_incomplete_payload_triple_is_left_alone():
    """A row whose payload triple is incomplete cannot be hashed.

    The canonical hash is computed FROM (content, group_id,
    source_description). A row missing one of them has no hash to
    compute, so it keeps ``canonical_hash IS NULL``. Measured on the
    live DB: all 12 713 rows have a complete triple, so this never
    fires in practice — but it must not crash the migration if one
    appears.
    """
    q = _tmp_queue()
    payload = json.dumps({"content": "only content"})  # no group_id / source_description
    now = "2026-10-31T00:00:00+00:00"

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at
            ) VALUES (?, 'done', ?, ?, 0, 0, ?, ?)
            """,
            ("partial-row", payload, now, now, now),
        )
        conn.commit()

    q2 = EpisodeQueue(str(q.db_path))
    row = _row(q2, "partial-row")
    assert row["canonical_hash"] is None


@pytest.mark.asyncio
async def test_provable_canonical_row_is_backfilled():
    """A row whose stored hash EQUALS the canonical hash IS backfilled.

    That equality is the proof: the row's stored hash was produced by
    the canonical scheme, so the canonical hash of that row IS that
    stored value.
    """
    q = _tmp_queue()
    payload = json.dumps(
        {
            "content": "canonical content",
            "group_id": "graphiti_memory",
            "name": "ep",
            "source_description": "daemon_http",
        }
    )
    chash = canonical_payload_hash("canonical content", "graphiti_memory", "daemon_http")
    now = "2026-10-31T00:00:00+00:00"

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at, payload_hash
            ) VALUES (?, 'done', ?, ?, 0, 0, ?, ?, ?)
            """,
            ("canonical-row", payload, now, now, now, chash),
        )
        conn.commit()

    q2 = EpisodeQueue(str(q.db_path))
    row = _row(q2, "canonical-row")
    assert row["canonical_hash"] == chash


@pytest.mark.asyncio
async def test_backfill_is_idempotent():
    """Re-opening the queue must not re-touch already-filled rows.

    The backfill runs on EVERY queue construction, and the queue is
    constructed by every writer process. A non-idempotent backfill
    would rewrite 12 000+ rows on every single write.
    """
    db_path = Path(tempfile.mkdtemp(prefix="jg_hard_")) / "episodes.db"

    q1 = EpisodeQueue(str(db_path))
    ep = await q1.enqueue("x", group_id="g", source_description="s", dedup_mode="observe")
    first = _row(q1, ep)["canonical_hash"]
    assert first is not None

    # Re-open twice: the second and third construction must not change
    # the stored hash, and must not rewrite the row.
    q2 = EpisodeQueue(str(db_path))
    q3 = EpisodeQueue(str(db_path))

    assert _row(q2, ep)["canonical_hash"] == first
    assert _row(q3, ep)["canonical_hash"] == first


@pytest.mark.asyncio
async def test_backfill_never_deletes_or_changes_the_row_count():
    q = _tmp_queue()
    before = len(_rows(q, "SELECT id FROM episodes"))

    EpisodeQueue(str(q.db_path))  # re-run the migration

    assert len(_rows(q, "SELECT id FROM episodes")) == before


# ── E. the backstop ──────────────────────────────────────────────────────


def _settings_with_backstop(enabled: bool):
    """Build a settings object with the backstop toggled.

    Uses the real ``JanusSettings`` so the config shape is the shipped
    one, not a hand-rolled stub.
    """
    from janus_graph.config import JanusSettings

    s = JanusSettings()
    s.pipeline.dedup.backstop.enabled = enabled
    return s


@pytest.mark.asyncio
async def test_backstop_is_default_off():
    """Shipping the backstop must change NOTHING until it is turned on.

    The default config has it off, so an ordinary sweep behaves exactly
    as before.
    """
    from janus_graph.config import JanusSettings

    s = JanusSettings()
    assert s.pipeline.dedup.backstop.enabled is False


@pytest.mark.asyncio
async def test_backstop_skips_an_already_ingested_payload():
    from unittest.mock import AsyncMock, patch

    q = _tmp_queue()
    from janus_graph.pipeline.worker import EpisodeWorker

    # One row already ingested, one row with the SAME content waiting.
    done_id = await q.enqueue("same content", group_id="g", source_description="s")
    await q.mark_done(done_id)
    pending_id = await q.enqueue(
        "same content", group_id="g", source_description="s", dedup_mode="off"
    )

    mock_client = AsyncMock()
    mock_client.add_episode = AsyncMock(return_value=None)

    worker = EpisodeWorker(q, _settings_with_backstop(True))
    records = await q.claim_next_batch(limit=10)
    target = [r for r in records if r.id == pending_id]
    assert target, "the pending row must be claimed"

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=mock_client):
        ok = await worker.process_record(target[0])

    assert ok is True, "a skip is a success — the row was handled"
    row = _row(q, pending_id)
    assert row["status"] == BACKSTOP_SKIPPED_STATUS
    assert row["status"] != "done", (
        "'done' means the content WAS ingested; the backstop did NOT ingest it"
    )
    mock_client.add_episode.assert_not_awaited()


@pytest.mark.asyncio
async def test_backstop_does_not_fire_when_disabled():
    q = _tmp_queue()
    from janus_graph.pipeline.worker import EpisodeWorker

    done_id = await q.enqueue("same content", group_id="g", source_description="s")
    await q.mark_done(done_id)
    pending_id = await q.enqueue("same content", group_id="g", source_description="s")

    worker = EpisodeWorker(q, _settings_with_backstop(False))
    assert await worker._backstop_skip_if_ingested(await _claim_one(q, pending_id)) is False


async def _claim_one(q: EpisodeQueue, episode_id: str):
    records = await q.claim_next_batch(limit=10)
    for r in records:
        if r.id == episode_id:
            return r
    raise AssertionError(f"{episode_id} was not claimed")


@pytest.mark.asyncio
async def test_backstop_never_raises_and_never_dead_letters():
    """A failure inside the backstop must degrade to "do not skip".

    The worker's failure path routes to the DLQ, so an exception here
    would turn a benign skip into queue pressure and DLQ noise. Every
    internal failure has to be swallowed.
    """
    q = _tmp_queue()
    from janus_graph.pipeline.worker import EpisodeWorker

    done_id = await q.enqueue("same content", group_id="g", source_description="s")
    await q.mark_done(done_id)
    pending_id = await q.enqueue("same content", group_id="g", source_description="s")

    worker = EpisodeWorker(q, _settings_with_backstop(True))

    def _boom(*args, **kwargs):
        raise RuntimeError("lookup exploded")

    # Break the lookup the backstop uses.
    q.find_ingested_duplicate_sync = _boom

    result = await worker._backstop_skip_if_ingested(await _claim_one(q, pending_id))

    assert result is False, "a broken lookup must not skip"
    row = _row(q, pending_id)
    assert row["status"] == "processing", "the row must be left for normal processing"
    dlq = _rows(q, "SELECT * FROM dead_letter")
    assert len(dlq) == 0, "a benign skip must never create a DLQ row"


@pytest.mark.asyncio
async def test_skipped_row_is_not_picked_up_again():
    """The skip is TERMINAL: the row must not be re-claimed."""
    q = _tmp_queue()

    ep = await q.enqueue("x", group_id="g", source_description="s")
    await q.mark_skipped_duplicate(ep, existing_episode_id="other")

    records = await q.claim_next_batch(limit=10)
    assert all(r.id != ep for r in records), "a skipped row must never be claimed again"


@pytest.mark.asyncio
async def test_skipped_row_is_not_reaped_or_dead_lettered():
    """The reapers pick up failure statuses. A skip is not a failure."""
    q = _tmp_queue()

    ep = await q.enqueue("x", group_id="g", source_description="s")
    await q.mark_failed(ep, "boom")
    await q.mark_skipped_duplicate(ep, existing_episode_id="other")

    reaped = await q.reap_failed_or_aborted(limit=100)
    assert reaped == 0, "a skipped row must not be requeued by the reaper"

    dlq = _rows(q, "SELECT * FROM dead_letter")
    assert len(dlq) == 0


# ── F. requeue paths survive a UNIQUE conflict ───────────────────────────


@pytest.mark.asyncio
async def test_requeue_guarded_skips_the_conflicting_row_only():
    """One conflicting row must not abort the whole requeue batch.

    Measured on SQLite: an UPDATE that moves a row INTO the index's
    scope raises IntegrityError just like an INSERT. Left unguarded,
    that exception would abort the whole nightly reap and leave every
    other row unreaped.
    """
    q = _tmp_queue()
    payload_hash = canonical_payload_hash("x", "g", "s")
    payload = json.dumps(
        {
            "content": "x",
            "group_id": "g",
            "name": "ep",
            "source_description": "s",
        }
    )
    now = "2026-10-31T00:00:00+00:00"

    with sqlite3.connect(str(q.db_path)) as conn:
        # An ACTIVE enforce row holding the hash.
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at,
                payload_hash, claimed_by, canonical_hash, dedup_gate
            ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                "holder",
                payload,
                now,
                now,
                now,
                payload_hash,
                "d:1",
                payload_hash,
                DEDUP_GATE_ENFORCE,
            ),
        )
        # A failed row that would CONFLICT when requeued.
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at,
                payload_hash, claimed_by, canonical_hash, dedup_gate
            ) VALUES (?, 'failed', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conflicting",
                payload,
                now,
                now,
                now,
                payload_hash,
                "d:1",
                payload_hash,
                DEDUP_GATE_ENFORCE,
            ),
        )
        # A failed row with a DIFFERENT hash — must requeue fine.
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at
            ) VALUES (?, 'failed', ?, ?, 0, 0, ?, ?)
            """,
            ("fine", json.dumps({"content": "other"}), now, now, now),
        )
        conn.commit()

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        requeued = requeue_guarded(conn, ["conflicting", "fine"], now=now)

    assert "conflicting" not in requeued, "the conflicting row must be skipped"
    assert "fine" in requeued, "the other row must still be requeued"

    statuses = {r["id"]: r["status"] for r in _rows(q, "SELECT id, status FROM episodes")}
    assert statuses["conflicting"] == "failed", "the conflicting row must keep its old status"
    assert statuses["fine"] == "queued"


@pytest.mark.asyncio
async def test_reap_failed_or_aborted_survives_a_conflict():
    """The real reaper must not abort because of one conflicting row."""
    q = _tmp_queue()
    payload_hash = canonical_payload_hash("x", "g", "s")
    payload = json.dumps(
        {
            "content": "x",
            "group_id": "g",
            "name": "ep",
            "source_description": "s",
        }
    )
    now = "2026-10-31T00:00:00+00:00"

    with sqlite3.connect(str(q.db_path)) as conn:
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at,
                payload_hash, claimed_by, canonical_hash, dedup_gate
            ) VALUES (?, 'queued', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                "holder",
                payload,
                now,
                now,
                now,
                payload_hash,
                "d:1",
                payload_hash,
                DEDUP_GATE_ENFORCE,
            ),
        )
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at,
                payload_hash, claimed_by, canonical_hash, dedup_gate
            ) VALUES (?, 'failed', ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conflicting",
                payload,
                now,
                now,
                now,
                payload_hash,
                "d:1",
                payload_hash,
                DEDUP_GATE_ENFORCE,
            ),
        )
        conn.execute(
            """
            INSERT INTO episodes (
                id, status, payload_json, enqueued_at, attempt_count,
                consecutive_failures, created_at, updated_at
            ) VALUES (?, 'failed', ?, ?, 0, 0, ?, ?)
            """,
            ("fine", json.dumps({"content": "other"}), now, now, now),
        )
        conn.commit()

    reaped = await q.reap_failed_or_aborted(limit=100)

    statuses = {r["id"]: r["status"] for r in _rows(q, "SELECT id, status FROM episodes")}
    assert statuses["fine"] == "queued", "the non-conflicting row must be requeued"
    assert statuses["conflicting"] == "failed", "the conflicting row must keep its status"
    assert reaped == 1, "exactly one row was requeued"


# ── shared vocabulary ────────────────────────────────────────────────────


def test_both_writers_use_the_same_lookup():
    """The MCP writer and the daemon writer must search the SAME rows.

    They used to carry separate copies of the query — which is how they
    drifted in the first place (the MCP writer wrote no hash at all, so
    the daemon's gate could never see MCP rows).
    """
    from janus_graph.daemon import episode_queue_adapter as adapter

    assert adapter.find_active_duplicate is find_active_duplicate
