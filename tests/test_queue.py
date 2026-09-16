"""Tests for SQLite WAL Episode Queue, DLQ, and retry policies."""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest
from janus_graph.pipeline.queue import EpisodeQueue, EpisodeRecord
from janus_graph.pipeline.retry import (
    compute_backoff_seconds,
    should_retry,
    send_to_dlq,
)


def _column_names(conn, table):
    conn.row_factory = sqlite3.Row
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _index_names(conn):
    conn.row_factory = sqlite3.Row
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }


def _tmp_dir():
    """Workaround for pytest tmp_path ownership issue (multi-user /tmp)."""
    return Path(tempfile.mkdtemp(prefix="jg_test_"))


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


# --- Schema migration idempotency tests (issue: cron_sweep crash post-3a68711) ---

PRE_REPLAY_SCHEMA = """
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
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_enqueued_at ON episodes(enqueued_at);

CREATE TABLE IF NOT EXISTS dead_letter (
    episode_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    last_error TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    failed_at TEXT NOT NULL,
    recovered_at TEXT,
    recovered_attempts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_dlq_failed_at ON dead_letter(failed_at);

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
"""


def _make_pre_replay_db(tmpdir):
    """Bootstrap a pre-3a68711 schema (no last_replay_at, no recovery indexes)."""
    db_path = os.path.join(tmpdir, "episodes.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(PRE_REPLAY_SCHEMA)
    conn.execute(
        "INSERT INTO episodes (id, payload_json, enqueued_at, created_at, updated_at)"
        " VALUES ('pre-existing', '{}', '2026-09-16T00:00:00Z',"
        " '2026-09-16T00:00:00Z', '2026-09-16T00:00:00Z')"
    )
    conn.commit()
    conn.close()
    return db_path


def test_init_db_migrates_pre_replay_schema():
    """EpisodeQueue.__init__ must idempotently add last_replay_at + 2 indexes."""
    tmp = _tmp_dir()
    db_path = _make_pre_replay_db(str(tmp))

    # Pre-condition: pre-3a68711 schema
    pre = sqlite3.connect(db_path)
    assert "last_replay_at" not in _column_names(pre, "episodes")
    pre_idx = _index_names(pre)
    assert "idx_last_replay_at" not in pre_idx
    assert "idx_dlq_recovered_at" not in pre_idx
    pre.close()

    # Boot EpisodeQueue → triggers _init_db_sync
    q = EpisodeQueue(db_path=db_path)

    # Post: schema advanced
    post = sqlite3.connect(db_path)
    cols = _column_names(post, "episodes")
    assert "last_replay_at" in cols
    post_idx = _index_names(post)
    assert "idx_last_replay_at" in post_idx
    assert "idx_dlq_recovered_at" in post_idx

    # Pre-existing row must still exist (no data loss)
    rows = post.execute("SELECT id FROM episodes WHERE id='pre-existing'").fetchall()
    assert len(rows) == 1
    post.close()


def test_init_db_idempotent_on_already_migrated():
    """Second boot must NOT raise — column exists, index exists, all no-ops."""
    tmp = _tmp_dir()
    db_path = _make_pre_replay_db(str(tmp))
    q1 = EpisodeQueue(db_path=db_path)  # migrates
    del q1

    # Snapshot row count
    pre = sqlite3.connect(db_path)
    n_before = pre.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    pre.close()

    # Reopen — second migration must be silent
    q2 = EpisodeQueue(db_path=db_path)
    del q2

    post = sqlite3.connect(db_path)
    cols = _column_names(post, "episodes")
    assert "last_replay_at" in cols
    assert "idx_last_replay_at" in _index_names(post)
    assert "idx_dlq_recovered_at" in _index_names(post)
    n_after = post.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert n_before == n_after
    post.close()


def test_init_db_fresh_db_no_op():
    """On fresh DB, CREATE TABLE IF NOT EXISTS branches handle schema; helpers skip."""
    tmp = _tmp_dir()
    db_path = os.path.join(str(tmp), "fresh.db")
    q = EpisodeQueue(db_path=db_path)
    del q

    post = sqlite3.connect(db_path)
    cols = _column_names(post, "episodes")
    assert "last_replay_at" in cols  # declared in SCHEMA
    assert "idx_last_replay_at" in _index_names(post)
    assert "idx_dlq_recovered_at" in _index_names(post)
    post.close()
