"""Tests for SQLite WAL Episode Queue, DLQ, and retry policies."""

import asyncio
import pytest
from datetime import datetime, timezone
from janus_graph.pipeline.queue import EpisodeQueue, EpisodeRecord
from janus_graph.pipeline.retry import (
    compute_backoff_seconds,
    should_retry,
    send_to_dlq,
)


@pytest.mark.asyncio
async def test_queue_enqueue_and_stats(temp_queue):
    stats = temp_queue.get_stats()
    assert stats.get("queued", 0) == 0
    assert stats.get("dlq", 0) == 0

    ep_id1 = await temp_queue.enqueue("Episode content 1", group_id="test_group")
    ep_id2 = await temp_queue.enqueue("Episode content 2", group_id="test_group")
    assert ep_id1 != ep_id2

    stats_after = temp_queue.get_stats()
    assert stats_after.get("queued") == 2


@pytest.mark.asyncio
async def test_queue_claim_and_mark_done(temp_queue):
    ep_id = await temp_queue.enqueue("Task payload", group_id="test_group")
    
    records = await temp_queue.claim_next_batch(limit=10)
    assert len(records) == 1
    assert records[0].id == ep_id
    assert records[0].content == "Task payload"
    assert records[0].status == "processing"

    await temp_queue.mark_done(ep_id)
    stats = temp_queue.get_stats()
    assert stats.get("done") == 1
    assert stats.get("queued", 0) == 0


@pytest.mark.asyncio
async def test_queue_mark_failed_and_abort(temp_queue):
    ep_id = await temp_queue.enqueue("Error payload", group_id="test_group")
    
    records = await temp_queue.claim_next_batch(limit=1)
    assert len(records) == 1

    await temp_queue.mark_failed(ep_id, "Temporary network timeout")
    stats = temp_queue.get_stats()
    assert stats.get("failed") == 1

    # Mark aborted -> moves to DLQ
    await temp_queue.mark_aborted(ep_id, "Fatal schema error")
    stats_aborted = temp_queue.get_stats()
    assert stats_aborted.get("aborted") == 1
    assert stats_aborted.get("dlq") == 1

    dlq_items = temp_queue.list_dlq(limit=10)
    assert len(dlq_items) == 1
    assert dlq_items[0]["episode_id"] == ep_id
    assert "Fatal schema error" in dlq_items[0]["last_error"]


@pytest.mark.asyncio
async def test_queue_dlq_replay(temp_queue):
    ep_id = await temp_queue.enqueue("DLQ replay item")
    await temp_queue.mark_aborted(ep_id, "Test error")
    
    assert temp_queue.get_stats().get("dlq") == 1
    replayed = await temp_queue.replay_dlq_episode(ep_id)
    assert replayed is True

    stats = temp_queue.get_stats()
    assert stats.get("queued") == 1


@pytest.mark.asyncio
async def test_queue_reap_failed_or_aborted(temp_queue):
    ep_id1 = await temp_queue.enqueue("Item 1")
    ep_id2 = await temp_queue.enqueue("Item 2")

    await temp_queue.mark_failed(ep_id1, "Fail 1")
    await temp_queue.mark_aborted(ep_id2, "Abort 2")

    reaped = await temp_queue.reap_failed_or_aborted(limit=10)
    assert reaped == 2

    stats = temp_queue.get_stats()
    assert stats.get("queued") == 2


@pytest.mark.asyncio
async def test_queue_replay_dlq_batch_filter_by_class(temp_queue):
    """Replay DLQ rows filtered by error class substring (case-insensitive)."""
    # Seed 3 DLQ rows: schema-drift, timeout, budget-exceeded
    ep_schema = await temp_queue.enqueue("schema drift item")
    ep_timeout = await temp_queue.enqueue("timeout item")
    ep_budget = await temp_queue.enqueue("budget item")

    await temp_queue.mark_aborted(ep_schema, "ValidationError: ExtractedEntities.extracted_entities Field required (SCHEMA_DRIFT)")
    await temp_queue.mark_aborted(ep_timeout, "TimeoutError: embed API > 30s (TIMEOUT)")
    await temp_queue.mark_aborted(ep_budget, "HTTP 402: budget exceeded $5/$5 (BUDGET_EXCEEDED)")

    assert temp_queue.get_stats().get("dlq") == 3

    # Replay only schema-drift + timeout, NOT budget-exceeded
    replayed = await temp_queue.replay_dlq_batch(
        limit=100,
        classes=["SCHEMA_DRIFT", "TIMEOUT"],
    )
    assert replayed == 2

    # Replayed rows → status='queued'; budget row stays in DLQ unrecovered.
    # get_stats().dlq counts rows WHERE recovered_at IS NULL, so it should drop to 1.
    stats_after = temp_queue.get_stats()
    assert stats_after.get("queued") == 2
    assert stats_after.get("dlq") == 1  # budget-exceeded still in DLQ

    # Verify the un-replayed row is the budget one — recovered_at is NULL on it,
    # non-NULL on the replayed ones.
    with temp_queue._get_connection() as conn:
        cur = conn.execute(
            "SELECT episode_id, recovered_at FROM dead_letter ORDER BY failed_at ASC"
        )
        rows = cur.fetchall()
    by_id = {r["episode_id"]: r["recovered_at"] for r in rows}
    assert by_id[ep_budget] is None
    assert by_id[ep_schema] is not None
    assert by_id[ep_timeout] is not None


@pytest.mark.asyncio
async def test_queue_replay_dlq_batch_no_filter_replays_all(temp_queue):
    """When classes=None, replay ALL unrecovered DLQ rows."""
    ep1 = await temp_queue.enqueue("item1")
    ep2 = await temp_queue.enqueue("item2")
    await temp_queue.mark_aborted(ep1, "any error A")
    await temp_queue.mark_aborted(ep2, "any error B")

    replayed = await temp_queue.replay_dlq_batch(limit=100)
    assert replayed == 2
    assert temp_queue.get_stats().get("dlq") == 0


@pytest.mark.asyncio
async def test_queue_replay_dlq_batch_respects_limit(temp_queue):
    """limit caps the number of rows replayed per call."""
    for i in range(5):
        ep_id = await temp_queue.enqueue(f"item{i}")
        await temp_queue.mark_aborted(ep_id, "SCHEMA_DRIFT error")

    assert temp_queue.get_stats().get("dlq") == 5
    replayed = await temp_queue.replay_dlq_batch(limit=3, classes=["SCHEMA_DRIFT"])
    assert replayed == 3
    assert temp_queue.get_stats().get("dlq") == 2  # 2 still pending
    assert temp_queue.get_stats().get("queued") == 3


@pytest.mark.asyncio
async def test_queue_replay_dlq_batch_idempotent(temp_queue):
    """Second call returns 0 — already-replayed rows have recovered_at set."""
    ep_id = await temp_queue.enqueue("item")
    await temp_queue.mark_aborted(ep_id, "SCHEMA_DRIFT")

    replayed = await temp_queue.replay_dlq_batch(limit=10, classes=["SCHEMA_DRIFT"])
    assert replayed == 1

    # Second call: row already recovered, should return 0
    replayed_again = await temp_queue.replay_dlq_batch(limit=10, classes=["SCHEMA_DRIFT"])
    assert replayed_again == 0


def test_retry_policies():
    assert compute_backoff_seconds(1) == 2.0
    assert compute_backoff_seconds(2) == 4.0
    assert compute_backoff_seconds(3) == 8.0

    record = EpisodeRecord(id="rec1", payload={"content": "test"}, attempt_count=0)
    
    # Transient error -> retryable
    assert should_retry(record, ConnectionError("Connection reset")) is True
    
    # Max attempts exceeded -> not retryable
    record.attempt_count = 3
    assert should_retry(record, ConnectionError("Connection reset"), max_attempts=3) is False

    # Fatal type error -> not retryable
    record.attempt_count = 0
    assert should_retry(record, TypeError("Missing argument")) is False
