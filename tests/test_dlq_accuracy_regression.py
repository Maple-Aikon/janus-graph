"""Regression tests for the 2026-10-12 DLQ-accuracy and error-message audit.

Each test here fails on the pre-fix code and passes on the fixed code. They are
written against observable behaviour (queue stats, stored strings), not against
implementation details, so a future refactor that keeps the contract green is
allowed while one that breaks it is not.

Background, measured on the live DB 2026-10-12:
  * ``/health`` reported ``dlq_open=341``, but 272 of those rows belonged to
    episodes whose status was already ``done`` - the dead_letter row was never
    closed when the episode finally succeeded.
  * 130 DLQ rows carried the literal string ``"TimeoutError: "`` with nothing
    after the colon, because ``str(asyncio.TimeoutError())`` is empty.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest

from janus_graph.pipeline.queue import EpisodeQueue, EpisodeRecord
from janus_graph.pipeline.retry import describe_exception, send_to_dlq
from janus_graph.pipeline.worker import EpisodeWorker


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """A real EpisodeQueue on a throwaway SQLite file."""
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(tmp_path))
    return EpisodeQueue(db_path=str(tmp_path / "episodes.db"))


def _dlq_rows(q: EpisodeQueue):
    with q._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        return {
            row["episode_id"]: dict(row)
            for row in conn.execute(
                "SELECT episode_id, recovered_at, recovered_attempts FROM dead_letter"
            ).fetchall()
        }


def _record(episode_id: str, attempt_count: int = 2) -> EpisodeRecord:
    return EpisodeRecord(
        id=episode_id,
        payload={"content": "x"},
        status="processing",
        enqueued_at="2026-10-12T00:00:00+00:00",
        started_at="2026-10-12T00:00:01+00:00",
        finished_at=None,
        attempt_count=attempt_count,
        consecutive_failures=0,
        last_error=None,
        checkpoint=None,
    )


# --------------------------------------------------------------------------
# Bug 1 - a successful episode must close its dead-letter row
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_done_closes_open_dlq_row(queue):
    """An episode that reaches 'done' must stop counting as an open DLQ row.

    Before the fix, ``mark_done`` only touched the ``episodes`` table, so the
    ``dead_letter`` row stayed ``recovered_at IS NULL`` forever and inflated
    every ``dlq``/``dlq_open`` count.
    """
    ep_id = await queue.enqueue("content that eventually succeeds")
    await queue.mark_aborted(ep_id, "TimeoutError: <no message>")
    assert queue.get_stats()["dlq"] == 1, "precondition: the row is open"

    await queue.mark_done(ep_id)

    assert queue.get_stats()["dlq"] == 0, "done episode must not stay an open DLQ row"
    rows = _dlq_rows(queue)
    assert rows[ep_id]["recovered_at"] is not None
    # The attempt count that caused the dead-lettering is preserved for forensics.
    assert rows[ep_id]["recovered_attempts"] == 1


@pytest.mark.asyncio
async def test_mark_done_leaves_closed_dlq_row_untouched(queue):
    """A second mark_done must not rewrite an already-closed DLQ row.

    Guards against a fix that blindly stamps recovered_at on every success,
    which would hide how many attempts actually failed.
    """
    ep_id = await queue.enqueue("content")
    await queue.mark_aborted(ep_id, "boom")
    await queue.mark_done(ep_id)
    first = _dlq_rows(queue)[ep_id]["recovered_at"]

    await queue.mark_done(ep_id)
    assert _dlq_rows(queue)[ep_id]["recovered_at"] == first


@pytest.mark.asyncio
async def test_mark_done_without_dlq_row_is_harmless(queue):
    """The common path - a successful first-try episode - must still work."""
    ep_id = await queue.enqueue("clean content")
    await queue.mark_done(ep_id)
    assert queue.get_stats().get("done") == 1
    assert _dlq_rows(queue) == {}


@pytest.mark.asyncio
async def test_replay_batch_does_not_requeue_a_done_episode(queue):
    """A stale DLQ row must not drag an already-ingested episode back to work.

    Before the fix, ``replay_dlq_batch`` flipped any episode back to 'queued'
    unconditionally. Re-ingesting an episode that already made it into the
    graph duplicates content, so the guard belongs in the replay path too.

    Note the precondition: the episode is 'done'. An episode that is merely
    'aborted' has not succeeded yet and MUST still be replayable - that is the
    entire purpose of the DLQ, and an over-broad guard would silently disable
    recovery.
    """
    ep_id = await queue.enqueue("content")
    await queue.mark_aborted(ep_id, "transient failure")
    await queue.mark_done(ep_id)
    # Simulate a pre-existing stale row: the episode completed but the DLQ
    # row was never closed (the exact state the 272 historical rows are in).
    with queue._get_connection() as conn:
        conn.execute("UPDATE dead_letter SET recovered_at = NULL WHERE episode_id = ?", (ep_id,))
        conn.commit()
    assert queue.get_stats()["dlq"] == 1

    replayed = await queue.replay_dlq_batch(limit=100)

    assert replayed == 0, "a done episode must not be counted as replayed"
    assert queue.get_stats().get("queued", 0) == 0, "must not be requeued"
    assert queue.get_stats().get("done") == 1, "must stay done"
    # The row is still closed, so it does not linger as phantom backlog.
    assert queue.get_stats()["dlq"] == 0


@pytest.mark.asyncio
async def test_replay_batch_still_replays_an_aborted_episode(queue):
    """The guard must not disable DLQ recovery.

    This is the negative control for the test above: 'aborted' means the work
    did NOT succeed, so replaying it is correct. A fix that rejected every
    non-queued episode would pass the previous test and break real recovery.
    """
    ep_id = await queue.enqueue("content that failed")
    await queue.mark_aborted(ep_id, "TimeoutError: <no message>")
    assert queue.get_stats().get("aborted") == 1

    replayed = await queue.replay_dlq_batch(limit=100)

    assert replayed == 1, "an aborted episode is exactly what the DLQ is for"
    assert queue.get_stats().get("queued") == 1


# --------------------------------------------------------------------------
# Bug 2 - an exception with no message must still say something useful
# --------------------------------------------------------------------------


def test_describe_exception_renders_missing_message():
    """str(asyncio.TimeoutError()) is empty - it must not produce 'X: '."""
    rendered = describe_exception(asyncio.TimeoutError())
    assert rendered == "TimeoutError: <no message>"


def test_describe_exception_keeps_real_messages():
    assert describe_exception(ValueError("bad json")) == "ValueError: bad json"


def test_describe_exception_keeps_replay_class_matching():
    """Replay filters LIKE-match last_error; the class name must survive.

    measured 2026-10-12: the old rendering also matched these classes, so this
    pins the contract that the fix did not silently change filtering.
    """
    rendered = describe_exception(asyncio.TimeoutError())
    for class_filter in ("timeout", "TimeoutError", "TIMEOUT"):
        assert class_filter.lower() in rendered.lower()


@pytest.mark.asyncio
async def test_dlq_timeout_row_is_identifiable(queue):
    """The end-to-end symptom: a timed-out episode must be readable later.

    Before the fix this wrote the literal 'TimeoutError: ' - 130 rows in the
    live DLQ, all indistinguishable.
    """
    ep_id = await queue.enqueue("content that times out")
    await send_to_dlq(queue, _record(ep_id), asyncio.TimeoutError())

    with queue._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT last_error FROM dead_letter WHERE episode_id = ?", (ep_id,)
        ).fetchone()

    assert row["last_error"] == "TimeoutError: <no message>"
    assert row["last_error"] != "TimeoutError: ", "regression: dangling colon"
    assert row["last_error"].strip() != "TimeoutError:", "regression: nothing recorded"


@pytest.mark.asyncio
async def test_worker_timeout_records_identifiable_error(queue, monkeypatch):
    """The worker's own write path must use the shared renderer.

    Before the fix, worker.py hand-rolled its own format for the retry path
    (``"TimeoutError: exceeded 300.0s"``) while send_to_dlq produced
    ``"TimeoutError: "`` for the identical failure. The same timeout therefore
    produced two different strings in two tables, so a LIKE filter for one could
    never find the other. This test pins the two writers to the SAME string.
    """
    worker = EpisodeWorker(queue, settings=_settings())
    # attempt_count=0 drives the mark_failed (retry) branch. With the default
    # of 2, should_retry() returns False and this test would silently exercise
    # the DLQ branch instead - the very branch that was never broken.
    record = _record(await queue.enqueue("slow content"), attempt_count=0)

    async def _hang(_record):
        await asyncio.sleep(10)

    monkeypatch.setattr(worker, "_execute_ingestion", _hang)

    ok = await worker.process_record(record, timeout_sec=0.01)

    assert ok is False
    assert queue.get_stats().get("failed") == 1, "precondition: took the retry path"
    stored = _stored_error(queue, "episodes", record.id)
    assert stored == "TimeoutError: <no message>", (
        "worker must render the timeout exactly like send_to_dlq does, "
        f"not with its own hand-rolled format: {stored!r}"
    )
    assert stored != "TimeoutError: ", "regression: dangling colon"
    assert "exceeded" not in stored, (
        "regression: the retry path and the DLQ path diverged again - a LIKE "
        "filter on one would never find the other"
    )


@pytest.mark.asyncio
async def test_worker_and_dlq_render_the_same_timeout_identically(queue, monkeypatch):
    """Cross-check: the retry path and the DLQ path must agree, byte for byte.

    This is the invariant that the fix actually establishes. The pre-fix code
    failed it while both individual strings still looked plausible.
    """
    worker = EpisodeWorker(queue, settings=_settings())

    async def _hang(_record):
        await asyncio.sleep(10)

    monkeypatch.setattr(worker, "_execute_ingestion", _hang)

    # 1. RETRY branch: attempt_count=0 -> should_retry True -> mark_failed
    ep_retry = await queue.enqueue("a")
    await worker.process_record(_record(ep_retry, attempt_count=0), timeout_sec=0.01)
    retry_string = _stored_error(queue, "episodes", ep_retry)
    assert queue.get_stats().get("failed") == 1, "precondition: took the retry path"

    # 2. DLQ branch: attempt_count=2 -> should_retry False -> send_to_dlq
    ep_dlq = await queue.enqueue("b")
    await worker.process_record(_record(ep_dlq, attempt_count=2), timeout_sec=0.01)
    with queue._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        dlq_string = conn.execute(
            "SELECT last_error FROM dead_letter WHERE episode_id = ?", (ep_dlq,)
        ).fetchone()["last_error"]
    assert queue.get_stats().get("aborted") == 1, "precondition: took the DLQ path"

    assert retry_string == dlq_string, (
        f"the two write paths disagree: episodes={retry_string!r} dead_letter={dlq_string!r}"
    )


def _stored_error(q: EpisodeQueue, table: str, episode_id: str) -> str:
    with q._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT last_error FROM {table} WHERE id = ?", (episode_id,)).fetchone()
    return row["last_error"]


def _settings():
    """Minimal settings object; EpisodeWorker only reads pipeline config."""
    from types import SimpleNamespace

    return SimpleNamespace(pipeline=SimpleNamespace(attempt_timeout_sec=600))
