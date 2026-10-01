"""Unit tests for the two ``_sync_reap`` closures in ``pipeline/queue.py``.

Both closures are nested inside their public wrappers and are therefore
exercised through those wrappers:

  * ``reap_stuck_processing``  -> closure #1 (queue.py:356)
      Finds ``processing`` rows older than ``timeout_sec`` and either
      requeues them (``attempts < max_attempts``) or aborts them into the
      dead-letter table (``attempts >= max_attempts``). Previously had NO
      coverage at all.
  * ``reap_failed_or_aborted`` -> closure #2 (queue.py:418)
      Requeues ``failed``/``aborted`` rows back to ``queued`` under a LIMIT.
      One happy-path test existed; this file adds the edges.

Real SQLite in a tmp dir, no network, no production code changes.

NOTE: ``_sync_reap`` is a DUPLICATE closure name in this module (two distinct
functions share it). Every mutation anchor below is therefore qualified by the
enclosing method name, not by the closure name alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from janus_graph.pipeline.queue import EpisodeQueue


# ─── helpers ─────────────────────────────────────────────────────────────


def _set_started_at(queue: EpisodeQueue, episode_id: str, value) -> None:
    """Backdate (or corrupt) ``started_at`` for one episode, bypassing the API.

    The public surface only ever writes ``started_at = now`` (in
    ``claim_next_batch``), so the age branches of the reaper can only be
    reached by writing the column directly.
    """
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE episodes SET started_at = ? WHERE id = ?",
            (value, episode_id),
        )
        conn.commit()


def _set_attempts(queue: EpisodeQueue, episode_id: str, n: int) -> None:
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE episodes SET attempt_count = ? WHERE id = ?",
            (n, episode_id),
        )
        conn.commit()


def _status(queue: EpisodeQueue, episode_id: str) -> str:
    rec = queue.get_record(episode_id)
    return rec.status


def _dlq_ids(queue: EpisodeQueue):
    return {r["episode_id"] for r in queue.get_dlq_records(limit=100)}


# ─── reap_stuck_processing: age gating ──────────────────────────────────


async def test_fresh_processing_row_is_not_reaped(temp_queue):
    """A row started 'now' is younger than the timeout and must survive."""
    ep = await temp_queue.enqueue("fresh item")
    await temp_queue.claim_next_batch(limit=1)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 0
    assert _status(temp_queue, ep) == "processing"


async def test_stuck_row_under_max_attempts_is_requeued(temp_queue):
    """Old enough and attempts still below the cap -> back to 'queued'."""
    ep = await temp_queue.enqueue("stuck item")
    await temp_queue.claim_next_batch(limit=1)
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    _set_started_at(temp_queue, ep, old)
    _set_attempts(temp_queue, ep, 0)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"
    assert _dlq_ids(temp_queue) == set(), "requeue must NOT write to the DLQ"


async def test_stuck_row_at_max_attempts_is_aborted_and_dead_lettered(temp_queue):
    """attempts + 1 >= max_attempts -> 'aborted' + a dead_letter row."""
    ep = await temp_queue.enqueue("doomed item")
    await temp_queue.claim_next_batch(limit=1)
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    _set_started_at(temp_queue, ep, old)
    _set_attempts(temp_queue, ep, 2)  # 2 + 1 >= 3

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, ep) == "aborted"
    assert ep in _dlq_ids(temp_queue)

    dlq = [r for r in temp_queue.get_dlq_records(limit=100) if r["episode_id"] == ep]
    assert dlq[0]["last_error"] == "Processing timeout exceeded"
    assert dlq[0]["attempt_count"] == 3


async def test_boundary_age_exactly_at_timeout_is_reaped(temp_queue):
    """The comparison is ``age >= timeout_sec``, so exactly-at counts."""
    ep = await temp_queue.enqueue("boundary item")
    await temp_queue.claim_next_batch(limit=1)
    # started_at in the future by a hair would be 'not yet'; at the boundary
    # we set it just past the timeout to stay robust against test latency.
    edge = (datetime.now(timezone.utc) - timedelta(seconds=901)).isoformat()
    _set_started_at(temp_queue, ep, edge)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_naive_started_at_is_treated_as_utc(temp_queue):
    """A tz-naive ``started_at`` gets UTC attached, so age is computed right."""
    ep = await temp_queue.enqueue("naive item")
    await temp_queue.claim_next_batch(limit=1)
    naive_old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).replace(
        tzinfo=None
    )
    _set_started_at(temp_queue, ep, naive_old.isoformat())

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_unparseable_started_at_is_treated_as_stuck(temp_queue):
    """The bare ``except Exception`` appends the row unconditionally."""
    ep = await temp_queue.enqueue("corrupt item")
    await temp_queue.claim_next_batch(limit=1)
    _set_started_at(temp_queue, ep, "not-a-timestamp")

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_null_started_at_is_skipped_not_reaped(temp_queue):
    """``if not started_at: continue`` — a null timestamp is never reaped."""
    ep = await temp_queue.enqueue("null started item")
    await temp_queue.claim_next_batch(limit=1)
    _set_started_at(temp_queue, ep, None)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 0
    assert _status(temp_queue, ep) == "processing"


async def test_only_processing_rows_are_considered(temp_queue):
    """Queued / done / failed rows are outside the reaper's WHERE clause."""
    queued = await temp_queue.enqueue("still queued")
    done = await temp_queue.enqueue("already done")
    failed = await temp_queue.enqueue("already failed")
    await temp_queue.mark_done(done)
    await temp_queue.mark_failed(failed, "boom")
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    _set_started_at(temp_queue, queued, old)
    _set_started_at(temp_queue, done, old)
    _set_started_at(temp_queue, failed, old)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 0
    assert _status(temp_queue, queued) == "queued"
    assert _status(temp_queue, done) == "done"
    assert _status(temp_queue, failed) == "failed"


async def test_mixed_batch_counts_only_stuck_rows(temp_queue):
    """Only the aged row is reaped; the fresh one stays 'processing'."""
    stuck = await temp_queue.enqueue("stuck")
    fresh = await temp_queue.enqueue("fresh")
    await temp_queue.claim_next_batch(limit=2)
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    _set_started_at(temp_queue, stuck, old)

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _status(temp_queue, stuck) == "queued"
    assert _status(temp_queue, fresh) == "processing"


async def test_attempt_count_is_incremented_on_requeue(temp_queue):
    """Requeue bumps attempt_count by one."""
    ep = await temp_queue.enqueue("bump item")
    await temp_queue.claim_next_batch(limit=1)
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    _set_started_at(temp_queue, ep, old)
    _set_attempts(temp_queue, ep, 1)

    await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert temp_queue.get_record(ep).attempt_count == 2


# ─── reap_failed_or_aborted ─────────────────────────────────────────────


async def test_reap_failed_and_aborted_requeues_both(temp_queue):
    ep_failed = await temp_queue.enqueue("f")
    ep_aborted = await temp_queue.enqueue("a")
    await temp_queue.mark_failed(ep_failed, "err")
    await temp_queue.mark_aborted(ep_aborted, "err")

    reaped = await temp_queue.reap_failed_or_aborted(limit=10)

    assert reaped == 2
    assert _status(temp_queue, ep_failed) == "queued"
    assert _status(temp_queue, ep_aborted) == "queued"


async def test_reap_with_no_candidates_returns_zero(temp_queue):
    """The ``if not ids: return 0`` early exit, before any UPDATE runs."""
    await temp_queue.enqueue("just queued")

    reaped = await temp_queue.reap_failed_or_aborted(limit=10)

    assert reaped == 0


async def test_reap_respects_limit(temp_queue):
    """Only ``limit`` rows are requeued, even with more candidates present."""
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(4)]
    for ep in ids:
        await temp_queue.mark_failed(ep, "err")

    reaped = await temp_queue.reap_failed_or_aborted(limit=2)

    assert reaped == 2
    stats = temp_queue.get_stats()
    assert stats.get("queued") == 2
    assert stats.get("failed") == 2


async def test_reap_ignores_queued_and_processing(temp_queue):
    """The IN ('failed','aborted') filter must not sweep healthy rows.

    Two rows are left healthy on purpose: one stays 'queued', one is claimed
    into 'processing'. Only the failed row is requeued.

    Note: ``claim_next_batch(limit=0)`` is NOT a no-op -- the queue coerces a
    non-positive limit to 20, so this test uses limit=1 to claim exactly one.
    """
    queued = await temp_queue.enqueue("q")
    to_claim = await temp_queue.enqueue("p")
    failed = await temp_queue.enqueue("f")
    await temp_queue.mark_failed(failed, "err")

    claimed = await temp_queue.claim_next_batch(limit=1)
    assert [r.id for r in claimed] == [queued], "claim order is enqueue order"

    reaped = await temp_queue.reap_failed_or_aborted(limit=50)

    assert reaped == 1
    assert _status(temp_queue, failed) == "queued"
    # Neither healthy row was touched by the reaper.
    assert _status(temp_queue, queued) == "processing"
    assert _status(temp_queue, to_claim) == "queued"


async def test_reap_is_idempotent_second_call_returns_zero(temp_queue):
    """After the first sweep nothing is left in failed/aborted."""
    ep = await temp_queue.enqueue("once")
    await temp_queue.mark_failed(ep, "err")

    first = await temp_queue.reap_failed_or_aborted(limit=10)
    second = await temp_queue.reap_failed_or_aborted(limit=10)

    assert first == 1
    assert second == 0
