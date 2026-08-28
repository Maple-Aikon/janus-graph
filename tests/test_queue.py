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
