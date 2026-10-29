"""NEGATIVE CONTROL — proves the OLD behaviour was actually WRONG.

These tests are written against the OLD public API only (no ``RetryPolicy``,
no ``_cron_payload``, no ``record_dream_run``), so they RUN on the
unpatched tree. On the unpatched tree they MUST fail; on the patched tree
they must pass.

A test that only proves "the new symbol does not exist yet" is a weak
signal — it proves the API changed, not that the behaviour was broken.
These tests assert the BEHAVIOUR:

  * the reaper requeued a PERMANENT 404 row (it had no filter at all),
  * the reaper requeued an EXHAUSTED row (it had no attempt ceiling),
  * the reaper did NOT refresh ``enqueued_at`` (the clause that closes the
    self-feeding loop),
  * the reaper picked the SAME rows on every call (the loop itself),
  * the manual replay re-armed a permanent 404 row AND reset its attempt
    counter (the third write path that bypassed the policy),
  * the manual replay did NOT refresh ``enqueued_at``.

Run with the patch stashed to see them fail; run with the patch applied to
see them pass. Both states are meaningful.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from janus_graph.pipeline.queue import EpisodeQueue


def _set_updated_at(queue: EpisodeQueue, episode_id: str, value: str) -> None:
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE episodes SET updated_at = ? WHERE id = ?",
            (value, episode_id),
        )
        conn.commit()


def _enqueued_at(queue: EpisodeQueue, episode_id: str) -> str:
    return queue.get_record(episode_id).enqueued_at


def _status(queue: EpisodeQueue, episode_id: str) -> str:
    return queue.get_record(episode_id).status


async def test_old_reaper_requeued_a_permanent_404_row(temp_queue):
    """OLD behaviour: the reaper had NO filter, so it requeued 404 rows.

    Measured 2026-10-23 on the live DB: 58 of the 100 rows requeued every
    night carried a permanent upstream fault (30 x 404, 28 x 402).
    """
    ep = await temp_queue.enqueue("permanent 404")
    await temp_queue.mark_failed(ep, "NotFoundError: Error code: 404 - model does not exist")

    reaped = await temp_queue.reap_failed_or_aborted(limit=10)

    assert reaped == 0, "the reaper must NOT requeue a permanent 404 row"


async def test_old_reaper_requeued_an_exhausted_row(temp_queue):
    """OLD behaviour: no attempt ceiling, so exhausted rows were requeued.

    Measured: 60 of the 100 nightly picks were already at
    ``attempt_count >= 3`` against ``max_attempts = 3``.
    """
    ep = await temp_queue.enqueue("exhausted")
    await temp_queue.mark_failed(ep, "transient error")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET attempt_count = 3 WHERE id = ?", (ep,))
        conn.commit()

    reaped = await temp_queue.reap_failed_or_aborted(limit=10)

    assert reaped == 0, "the reaper must NOT requeue a row at the attempt ceiling"


async def test_old_reaper_did_not_refresh_enqueued_at(temp_queue):
    """OLD behaviour: ``enqueued_at`` was left at its original value.

    That is the clause that closes the self-feeding loop: with the old
    timestamp, ``claim_next_batch`` (ORDER BY ``enqueued_at`` ASC) handed
    the row straight back to the worker, which re-failed it, and the next
    night's reap picked it again.
    """
    ep = await temp_queue.enqueue("old head")
    await temp_queue.mark_failed(ep, "transient error")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET enqueued_at = ? WHERE id = ?", (stale, ep))
        conn.commit()

    await temp_queue.reap_failed_or_aborted(limit=10)

    assert _enqueued_at(temp_queue, ep) != stale, "enqueued_at MUST be refreshed on requeue"


async def test_old_reaper_picked_the_same_rows_every_time(temp_queue):
    """OLD behaviour: the pick-set did not advance.

    Without ``ORDER BY updated_at ASC`` SQLite scanned ``idx_status`` and
    returned the same rows every time — the same 100 rows every night since
    2026-10-09, with 5,228 of 5,328 rows never requeued at all.
    """
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(6)]
    base = datetime.now(timezone.utc)
    for i, ep in enumerate(ids):
        await temp_queue.mark_failed(ep, "transient error")
        _set_updated_at(
            temp_queue,
            ep,
            (base - timedelta(minutes=60 - i)).isoformat(),
        )

    await temp_queue.reap_failed_or_aborted(limit=3)
    picked_first = {ep for ep in ids if _status(temp_queue, ep) == "queued"}

    # Force the first batch back to failed so the second reap has to choose.
    for ep in picked_first:
        await temp_queue.mark_failed(ep, "transient error")

    await temp_queue.reap_failed_or_aborted(limit=3)
    picked_second = {ep for ep in ids if _status(temp_queue, ep) == "queued"}

    assert picked_first != picked_second, "the second reap must pick DIFFERENT rows"


async def test_old_manual_replay_rearmed_a_permanent_404_row(temp_queue):
    """OLD behaviour: the manual replay had no filter and reset attempts.

    Measured: 250 episodes carry ``last_replay_at`` and 27 of them were
    still failed/aborted with a permanent 404/402 code — a manual retry
    could re-arm a permanent upstream fault and make the cycle repeatable
    by hand.
    """
    ep = await temp_queue.enqueue("permanent")
    await temp_queue.mark_aborted(ep, "NotFoundError: Error code: 404 - no such model")

    ok = await temp_queue.replay_dlq_episode(ep)

    assert ok is False, "the manual replay must REFUSE a permanent 404 row"
    assert _status(temp_queue, ep) == "aborted"


async def test_old_manual_replay_did_not_refresh_enqueued_at(temp_queue):
    """OLD behaviour: the manual replay left ``enqueued_at`` stale."""
    ep = await temp_queue.enqueue("replayed")
    await temp_queue.mark_aborted(ep, "TimeoutError: slow")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET enqueued_at = ? WHERE id = ?", (stale, ep))
        conn.commit()

    await temp_queue.replay_dlq_episode(ep)

    assert _enqueued_at(temp_queue, ep) != stale, "enqueued_at MUST be refreshed on replay"


async def test_old_reaper_default_limit_was_500(temp_queue):
    """OLD behaviour: the signature default was ``limit=500`` with no policy.

    A default that silently means "reap 500 rows with no policy" IS the
    bug, so the default now comes from the shared policy (24).
    """
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(30)]
    for ep in ids:
        await temp_queue.mark_failed(ep, "transient error")

    reaped = await temp_queue.reap_failed_or_aborted()

    assert reaped <= 24, "the default limit must come from the policy (24), not 500"
