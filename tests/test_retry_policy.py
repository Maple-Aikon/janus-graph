"""Tests for the SHARED retry policy and the reap starvation fix.

Scope (plan ``dream-mode-nightly-hardening-...-20261023``, T1 + T2):
  * ONE policy, THREE write paths — ``reap_failed_or_aborted``,
    ``replay_dlq_batch`` and ``replay_dlq_episode`` must all obey the same
    denylist. Before this, the reaper had NO filter at all, the batch replay
    used a narrow allowlist, and the manual replay had no filter AND reset
    ``attempt_count`` to 0 (so a manual retry could re-arm a permanent
    upstream fault and make the cycle repeatable by hand).
  * The reap pick-set must ADVANCE (it used to be the same 100 rows every
    night — a self-feeding loop).
  * ``enqueued_at`` must be refreshed on requeue (the clause that closes
    that loop).
  * The attempt ceiling must be respected (60 of the 100 nightly picks were
    already exhausted).
  * The DLQ-skip must apply to ``failed`` rows ONLY (``mark_aborted`` writes
    a DLQ row, so a blanket skip would exclude every aborted episode).
  * The denylist predicate must be NULL-safe (a plain ``NOT LIKE`` evaluates
    to NULL for a NULL column and DROPS the row).

Real SQLite in a tmp dir, no network, no production data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from janus_graph.pipeline.queue import EpisodeQueue, RetryPolicy, _policy_where_clause

# ─── helpers ─────────────────────────────────────────────────────────────


def _set_updated_at(queue: EpisodeQueue, episode_id: str, value: str) -> None:
    with queue._get_connection() as conn:
        conn.execute(
            "UPDATE episodes SET updated_at = ? WHERE id = ?",
            (value, episode_id),
        )
        conn.commit()


def _enqueued_at(queue: EpisodeQueue, episode_id: str) -> str:
    rec = queue.get_record(episode_id)
    return rec.enqueued_at


def _status(queue: EpisodeQueue, episode_id: str) -> str:
    return queue.get_record(episode_id).status


def _attempt_count(queue: EpisodeQueue, episode_id: str) -> int:
    return queue.get_record(episode_id).attempt_count


def _consecutive_failures(queue: EpisodeQueue, episode_id: str) -> int:
    return queue.get_record(episode_id).consecutive_failures


def _last_error(queue: EpisodeQueue, episode_id: str):
    return queue.get_record(episode_id).last_error


# ─── T1: the shared policy ───────────────────────────────────────────────


def test_default_policy_denies_the_two_measured_permanent_codes():
    """The denylist default is the MEASURED permanent set, not a guess."""
    policy = RetryPolicy()
    assert "Error code: 404" in policy.permanent_error_codes
    assert "Error code: 402" in policy.permanent_error_codes
    assert policy.reap_limit == 24
    assert policy.respect_attempt_ceiling is True


def test_policy_from_settings_reads_the_config_section():
    """``from_settings`` picks up a configured policy."""

    class _Policy:
        permanent_error_codes = ["boom"]
        allowed_error_codes = []
        reap_limit = 7
        respect_attempt_ceiling = False

    class _Pipeline:
        retry_policy = _Policy()
        max_attempts = 5

    class _Settings:
        pipeline = _Pipeline()

    policy = RetryPolicy.from_settings(_Settings())
    assert policy.permanent_error_codes == ("boom",)
    assert policy.reap_limit == 7
    assert policy.respect_attempt_ceiling is False
    assert policy.max_attempts == 5


def test_policy_from_settings_survives_a_settings_object_without_the_section():
    """``contracts.Settings`` has no ``pipeline.retry_policy`` at all.

    A direct attribute chain would raise exactly when the guard was needed
    — the same defect class as ``cron.py:37-43``.
    """

    class _Settings:
        pass

    policy = RetryPolicy.from_settings(_Settings())
    assert policy.reap_limit == 24
    assert policy.permanent_error_codes == ("Error code: 404", "Error code: 402")


# ─── T1/T2: the reaper obeys the policy ──────────────────────────────────


async def test_reap_skips_a_row_with_a_permanent_404_code(temp_queue):
    """A 404 row is a PERMANENT upstream fault — never requeued."""
    ep = await temp_queue.enqueue("permanent 404")
    await temp_queue.mark_failed(ep, "NotFoundError: Error code: 404 - model does not exist")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


async def test_reap_skips_a_row_with_a_permanent_402_code(temp_queue):
    """A 402 row means the upstream budget is exhausted — never requeued."""
    ep = await temp_queue.enqueue("permanent 402")
    await temp_queue.mark_failed(ep, "APIStatusError: Error code: 402 - budget exhausted")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


async def test_reap_keeps_a_transient_500_row(temp_queue):
    """``InternalServerError: Error code: 500`` is transient — it IS requeued."""
    ep = await temp_queue.enqueue("transient 500")
    await temp_queue.mark_failed(ep, "InternalServerError: Error code: 500 - upstream hiccup")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_reap_keeps_a_connection_error_row(temp_queue):
    """``ConnectionError`` (local dependency down) is recoverable — requeued."""
    ep = await temp_queue.enqueue("connection error")
    await temp_queue.mark_failed(ep, "ConnectionError: Error 111 connecting to 127.0.0.1:6379")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_reap_keeps_a_row_with_a_null_last_error(temp_queue):
    """NULL-safety: a row we CANNOT classify must NOT be dropped.

    The policy's own rule is "a row with no ``last_error`` IS requeued; do
    not drop rows you cannot classify". A bare ``NOT LIKE`` predicate would
    evaluate to NULL for a NULL column and exclude the row — the exact
    opposite. This is the contract.
    """
    ep = await temp_queue.enqueue("null error")
    await temp_queue.mark_failed(ep, "transient error")
    # mark_failed always writes an error, so clear it directly.
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET last_error = NULL WHERE id = ?", (ep,))
        conn.commit()

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_reap_policy_matching_is_case_insensitive(temp_queue):
    """The code match is case-insensitive, like the DLQ class predicate."""
    ep = await temp_queue.enqueue("lowercase 404")
    await temp_queue.mark_failed(ep, "notfounderror: error CODE: 404 - nope")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


async def test_reap_policy_code_is_not_treated_as_a_like_wildcard(temp_queue):
    """A configured code containing ``_`` must not match every row.

    ``LIKE '%_%'`` matches every non-empty error (measured 2026-10-16 on the
    live DB: 536/536 rows vs 375/536 for a literal underscore). The policy
    therefore uses ``instr()``, which has no wildcard semantics.
    """
    ep = await temp_queue.enqueue("underscore code")
    await temp_queue.mark_failed(ep, "SomeError: the quick brown fox")

    policy = RetryPolicy(permanent_error_codes=("Error_code",))
    reaped = await temp_queue.reap_failed_or_aborted(policy=policy)

    assert reaped == 1, "a literal '_' must not behave as a LIKE wildcard"


async def test_reap_allowlist_wins_over_the_denylist(temp_queue):
    """An explicit allowlist takes precedence over the permanent denylist."""
    allowed = await temp_queue.enqueue("allowlisted")
    denied = await temp_queue.enqueue("not allowlisted")
    await temp_queue.mark_failed(allowed, "SomeError: Error code: 500")
    await temp_queue.mark_failed(denied, "SomeError: Error code: 999")

    policy = RetryPolicy(
        permanent_error_codes=("Error code: 404",),
        allowed_error_codes=("Error code: 500",),
    )
    reaped = await temp_queue.reap_failed_or_aborted(policy=policy)

    assert reaped == 1
    assert _status(temp_queue, allowed) == "queued"
    assert _status(temp_queue, denied) == "failed"


# ─── T2: the attempt ceiling ─────────────────────────────────────────────


async def test_reap_respects_the_attempt_ceiling(temp_queue):
    """A row that already exhausted its attempts is provably futile work.

    Measured 2026-10-23: 60 of the 100 rows the reaper picked every night
    had ``attempt_count >= 3`` against ``max_attempts = 3``.
    """
    ep = await temp_queue.enqueue("exhausted")
    await temp_queue.mark_failed(ep, "transient error")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET attempt_count = 3 WHERE id = ?", (ep,))
        conn.commit()

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


async def test_reap_can_ignore_the_attempt_ceiling_when_configured(temp_queue):
    """The ceiling is a policy switch, not a hard-coded rule."""
    ep = await temp_queue.enqueue("exhausted but allowed")
    await temp_queue.mark_failed(ep, "transient error")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET attempt_count = 3 WHERE id = ?", (ep,))
        conn.commit()

    policy = RetryPolicy(respect_attempt_ceiling=False)
    reaped = await temp_queue.reap_failed_or_aborted(policy=policy)

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


async def test_reap_does_not_reset_attempt_count(temp_queue):
    """``attempt_count`` is the escalation signal read by ``should_retry``.

    Resetting it on requeue would make the DLQ escalation never trigger
    (``retry.py:29`` reads ``record.attempt_count + 1 >= max_attempts``).
    """
    ep = await temp_queue.enqueue("keeps its attempts")
    await temp_queue.mark_failed(ep, "transient error")
    before = _attempt_count(temp_queue, ep)

    await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert _attempt_count(temp_queue, ep) == before


async def test_reap_resets_consecutive_failures(temp_queue):
    """``consecutive_failures`` is reset on requeue.

    It is a DEAD counter (written in exactly one place, read in zero), so
    this changes no decision — but it makes the column mean what its name
    says.
    """
    ep = await temp_queue.enqueue("streak")
    await temp_queue.mark_failed(ep, "transient error")
    await temp_queue.mark_failed(ep, "transient error")
    assert _consecutive_failures(temp_queue, ep) == 2

    await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert _consecutive_failures(temp_queue, ep) == 0


async def test_mark_done_resets_consecutive_failures(temp_queue):
    """``mark_done`` also resets the streak."""
    ep = await temp_queue.enqueue("done")
    await temp_queue.mark_failed(ep, "transient error")
    assert _consecutive_failures(temp_queue, ep) == 1

    await temp_queue.mark_done(ep)

    assert _consecutive_failures(temp_queue, ep) == 0


# ─── T2: the DLQ-skip asymmetry ──────────────────────────────────────────


async def test_reap_skips_a_failed_row_that_has_an_open_dlq_row(temp_queue):
    """A ``failed`` row with an open DLQ row is owned by ``replay_dlq_batch``.

    The DLQ row is the OWNERSHIP marker: exactly one of the two paths owns
    each row, so they must not both requeue it.
    """
    ep = await temp_queue.enqueue("failed with dlq row")
    await temp_queue.mark_failed(ep, "transient error")
    # mark_aborted writes the dead_letter row; force the episode back to
    # 'failed' so it is the failed-with-open-DLQ case.
    await temp_queue.mark_aborted(ep, "transient error")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET status = 'failed' WHERE id = ?", (ep,))
        conn.commit()

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


async def test_reap_still_requeues_an_aborted_row_with_an_open_dlq_row(temp_queue):
    """An ``aborted`` row's DLQ row is its TERMINAL state, not an ownership claim.

    This is the asymmetry, stated so it is not "fixed away" later: a blanket
    "skip rows with an open DLQ row" would exclude ALL aborted rows.
    """
    ep = await temp_queue.enqueue("aborted with dlq row")
    await temp_queue.mark_aborted(ep, "transient error")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert reaped == 1
    assert _status(temp_queue, ep) == "queued"


# ─── T2: the fairness fix — the pick-set must ADVANCE ────────────────────


async def test_reap_picks_the_oldest_updated_at_first(temp_queue):
    """The pick is ordered by ``updated_at ASC``.

    Without the ORDER BY, SQLite scanned ``idx_status`` and returned the
    same rows every time — the self-feeding 100-row loop.
    """
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(3)]
    base = datetime.now(timezone.utc)
    for i, ep in enumerate(ids):
        await temp_queue.mark_failed(ep, "transient error")
        # ids[0] is the OLDEST updated_at.
        _set_updated_at(
            temp_queue,
            ep,
            (base - timedelta(minutes=10 - i)).isoformat(),
        )

    reaped = await temp_queue.reap_failed_or_aborted(limit=1, policy=RetryPolicy())

    assert reaped == 1
    assert _status(temp_queue, ids[0]) == "queued"
    assert _status(temp_queue, ids[1]) == "failed"


async def test_second_reap_picks_different_rows_than_the_first(temp_queue):
    """The second reap must NOT pick the rows the first one picked.

    This is the regression test for the self-feeding loop: under the old
    code the same rows were requeued every night, forever.
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

    first = await temp_queue.reap_failed_or_aborted(limit=3, policy=RetryPolicy())
    assert first == 3
    picked_first = {ep for ep in ids if _status(temp_queue, ep) == "queued"}
    assert len(picked_first) == 3

    # The requeued rows now have a FRESH updated_at (the reaper writes it),
    # so the next reap must move on to the untouched rows.
    second = await temp_queue.reap_failed_or_aborted(limit=3, policy=RetryPolicy())
    assert second == 3
    picked_second = {ep for ep in ids if _status(temp_queue, ep) == "queued"}
    assert picked_second == set(ids)
    assert picked_first < picked_second, "the second reap must advance the pick-set"


async def test_reap_refreshes_enqueued_at_on_requeue(temp_queue):
    """``enqueued_at`` MUST be refreshed — this is the load-bearing clause.

    Without it, the reaped rows keep the oldest ``enqueued_at`` in the
    table, and ``claim_next_batch`` (ORDER BY ``enqueued_at`` ASC) hands them
    straight back to the worker, which re-fails them, and the next night's
    reap picks them again.
    """
    ep = await temp_queue.enqueue("old head")
    await temp_queue.mark_failed(ep, "transient error")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET enqueued_at = ? WHERE id = ?", (stale, ep))
        conn.commit()
    assert _enqueued_at(temp_queue, ep) == stale

    await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    assert _enqueued_at(temp_queue, ep) != stale
    assert _enqueued_at(temp_queue, ep) > stale


async def test_requeued_row_does_not_outrank_a_newer_episode(temp_queue):
    """A requeued row must not jump ahead of a genuinely new episode.

    This is the queue-head fairness property the whole fix is about.
    """
    old = await temp_queue.enqueue("old")
    await temp_queue.mark_failed(old, "transient error")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET enqueued_at = ? WHERE id = ?", (stale, old))
        conn.commit()

    new = await temp_queue.enqueue("new")

    await temp_queue.reap_failed_or_aborted(policy=RetryPolicy())

    claimed = await temp_queue.claim_next_batch(limit=1)
    assert [r.id for r in claimed] == [new], "the new episode must be claimed first"


async def test_reap_stuck_processing_refreshes_enqueued_at(temp_queue):
    """``reap_stuck_processing`` is a FOURTH writer of ``status='queued'``.

    It runs on every sweep tick, so leaving ``enqueued_at`` stale made it a
    live second feeder of the queue head.
    """
    ep = await temp_queue.enqueue("stuck")
    await temp_queue.claim_next_batch(limit=1)
    old = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute(
            "UPDATE episodes SET started_at = ?, enqueued_at = ? WHERE id = ?",
            (old, stale, ep),
        )
        conn.commit()

    reaped = await temp_queue.reap_stuck_processing(timeout_sec=900, max_attempts=3)

    assert reaped == 1
    assert _enqueued_at(temp_queue, ep) != stale
    assert _enqueued_at(temp_queue, ep) > stale


# ─── T2: the limit comes from the policy ─────────────────────────────────


async def test_reap_limit_defaults_to_the_policy_value(temp_queue):
    """The signature default used to be ``limit=500`` with NO policy.

    A default that silently means "reap 500 rows with no policy" IS the
    bug, so the default now comes from the shared policy.
    """
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(30)]
    for ep in ids:
        await temp_queue.mark_failed(ep, "transient error")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy(reap_limit=5))

    assert reaped == 5


async def test_reap_limit_zero_is_a_no_op(temp_queue):
    """``reap_limit: 0`` is the documented rollback: one config line."""
    ep = await temp_queue.enqueue("item")
    await temp_queue.mark_failed(ep, "transient error")

    reaped = await temp_queue.reap_failed_or_aborted(policy=RetryPolicy(reap_limit=0))

    assert reaped == 0
    assert _status(temp_queue, ep) == "failed"


# ─── T2 item 9: the manual replay obeys the policy too ──────────────────


async def test_replay_dlq_episode_refuses_a_permanent_code_row(temp_queue):
    """A manual retry must not be able to re-arm a permanent upstream fault.

    The old path wrote ``status='queued', attempt_count = 0`` for ONE row
    with NO filter — so an operator could reset the attempt counter of a
    404/402 row by hand and make the cycle repeatable.
    """
    ep = await temp_queue.enqueue("permanent")
    await temp_queue.mark_aborted(ep, "NotFoundError: Error code: 404 - no such model")

    ok = await temp_queue.replay_dlq_episode(ep, policy=RetryPolicy())

    assert ok is False
    assert _status(temp_queue, ep) == "aborted"
    assert _attempt_count(temp_queue, ep) == 1, "attempt_count must NOT be reset"


async def test_replay_dlq_episode_refusal_names_the_offending_code(temp_queue):
    """The refusal must NAME the code — "refused" alone teaches nothing."""
    ep = await temp_queue.enqueue("permanent")
    await temp_queue.mark_aborted(ep, "NotFoundError: Error code: 404 - no such model")

    reason = temp_queue.dlq_refusal_reason(ep, RetryPolicy())

    assert reason is not None
    assert "Error code: 404" in reason


async def test_replay_dlq_episode_allows_a_transient_row(temp_queue):
    """A transient row is still replayable by hand."""
    ep = await temp_queue.enqueue("transient")
    await temp_queue.mark_aborted(ep, "TimeoutError: embed API > 30s")

    ok = await temp_queue.replay_dlq_episode(ep, policy=RetryPolicy())

    assert ok is True
    assert _status(temp_queue, ep) == "queued"
    assert _attempt_count(temp_queue, ep) == 0


async def test_replay_dlq_episode_refreshes_enqueued_at(temp_queue):
    """A replayed row must not come back as the queue head."""
    ep = await temp_queue.enqueue("replayed")
    await temp_queue.mark_aborted(ep, "TimeoutError: slow")
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET enqueued_at = ? WHERE id = ?", (stale, ep))
        conn.commit()

    await temp_queue.replay_dlq_episode(ep, policy=RetryPolicy())

    assert _enqueued_at(temp_queue, ep) != stale


async def test_replay_dlq_batch_applies_the_shared_denylist(temp_queue):
    """The batch replay and the reaper must not disagree.

    The batch replay's ``classes`` allowlist stays as a NARROWER second
    filter; the denylist is the shared FLOOR.
    """
    permanent = await temp_queue.enqueue("permanent")
    transient = await temp_queue.enqueue("transient")
    await temp_queue.mark_aborted(permanent, "NotFoundError: Error code: 404 - nope")
    await temp_queue.mark_aborted(transient, "TimeoutError: slow")

    replayed = await temp_queue.replay_dlq_batch(
        limit=100,
        classes=["TimeoutError"],
        policy=RetryPolicy(),
    )

    assert replayed == 1
    assert _status(temp_queue, transient) == "queued"
    assert _status(temp_queue, permanent) == "aborted"


async def test_replay_dlq_batch_policy_predicate_is_null_safe(temp_queue):
    """The DLQ predicate must be NULL-safe too.

    ``dead_letter.last_error`` is declared NOT NULL in the schema, so a
    NULL row cannot be produced through the public API — but the predicate
    must still be written NULL-safe, so that a future schema change (or a
    hand-edited row) cannot silently drop rows the policy cannot classify.

    The predicate is asserted directly rather than through a crafted NULL
    row: crafting one would require dropping the schema's NOT NULL
    constraint, which is a far bigger change than the property being
    tested.
    """
    clauses, params = _policy_where_clause(RetryPolicy(), "last_error")

    assert clauses, "the denylist must produce a predicate"
    for clause in clauses:
        assert "IS NULL" in clause, f"predicate must be NULL-safe: {clause}"
    assert len(params) == 2

    # The predicate evaluated against a NULL column must NOT exclude the
    # row. ``episodes.last_error`` accepts NULL (only ``dead_letter``'s is
    # NOT NULL), so a NULL row can be seeded here and the predicate applied
    # to it directly. The classic ``NOT LIKE`` trap drops it; the NULL-safe
    # form keeps it.
    ep = await temp_queue.enqueue("null probe")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET last_error = NULL WHERE id = ?", (ep,))
        conn.commit()
        found = conn.execute(
            "SELECT id FROM episodes WHERE id = ? AND (" + " AND ".join(clauses) + ")",
            [ep] + params,
        ).fetchone()
    assert found is not None, "a NULL last_error row must pass the predicate"


# ─── T4: the fairness signal ─────────────────────────────────────────────


async def test_reap_fairness_stats_report_the_repeat_rate(temp_queue):
    """The repeat rate is the signal that would have exposed the loop.

    Under the old code it was 100 % — the same 100 rows every night.
    """
    ids = [await temp_queue.enqueue(f"item {i}") for i in range(4)]
    for ep in ids:
        await temp_queue.mark_failed(ep, "transient error")

    stats = temp_queue.reap_fairness_stats(max_attempts=3)

    assert stats["eligible"] == 4
    assert stats["exhausted"] == 0
    assert stats["repeated_turns"] == 0
    assert stats["repeat_rate"] == 0.0


async def test_reap_fairness_stats_count_exhausted_rows(temp_queue):
    """Rows at the ceiling can never be requeued — they are reported apart."""
    ep = await temp_queue.enqueue("exhausted")
    await temp_queue.mark_failed(ep, "transient error")
    with temp_queue._get_connection() as conn:
        conn.execute("UPDATE episodes SET attempt_count = 3 WHERE id = ?", (ep,))
        conn.commit()

    stats = temp_queue.reap_fairness_stats(max_attempts=3)

    assert stats["exhausted"] == 1


async def test_reap_fairness_stats_are_defined_on_an_empty_queue(temp_queue):
    """The number must always be defined, never a ZeroDivisionError."""
    stats = temp_queue.reap_fairness_stats(max_attempts=3)

    assert stats["eligible"] == 0
    assert stats["repeat_rate"] == 0.0
