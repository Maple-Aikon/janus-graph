"""Tests for Pipeline Worker, Cron Sweeper, and Dream Mode Consolidation."""

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from janus_graph.core.contracts import PipelineSettings, ReportSettings, Settings
from janus_graph.pipeline.cron import run_cron_sweep
from janus_graph.pipeline.dream import (
    bounded_label_propagation,
    run_dream_consolidation,
)
from janus_graph.pipeline.queue import EpisodeQueue, EpisodeRecord
from janus_graph.pipeline.worker import EpisodeWorker


@pytest.mark.asyncio
async def test_worker_process_empty_content(temp_queue):
    ep_id = await temp_queue.enqueue("")
    records = await temp_queue.claim_next_batch(1)
    assert len(records) == 1

    worker = EpisodeWorker(temp_queue)
    ok = await worker.process_record(records[0])
    assert ok is True
    stats = temp_queue.get_stats()
    assert stats.get("done") == 1


@pytest.mark.asyncio
async def test_worker_process_success(temp_queue):
    ep_id = await temp_queue.enqueue("Some memory fact", group_id="custom_grp")
    records = await temp_queue.claim_next_batch(1)
    assert len(records) == 1

    worker = EpisodeWorker(temp_queue)

    # Mock Graphiti client
    mock_client = AsyncMock()
    mock_client.add_episode = AsyncMock(return_value=None)

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=mock_client):
        ok = await worker.process_record(records[0])
        assert ok is True
        mock_client.add_episode.assert_awaited_once_with(
            name=f"ep_{ep_id[:8]}",
            episode_body="Some memory fact",
            source_description="agent_interaction",
            reference_time=ANY,
            group_id="custom_grp",
        )

    stats = temp_queue.get_stats()
    assert stats.get("done") == 1


@pytest.mark.asyncio
async def test_worker_process_timeout(temp_queue):
    ep_id = await temp_queue.enqueue("Slow memory fact")
    records = await temp_queue.claim_next_batch(1)

    worker = EpisodeWorker(temp_queue)

    async def _slow_add(*args, **kwargs):
        await asyncio.sleep(0.5)

    mock_client = MagicMock()
    mock_client.add_episode = _slow_add

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=mock_client):
        ok = await worker.process_record(records[0], timeout_sec=0.05)
        assert ok is False

    stats = temp_queue.get_stats()
    assert stats.get("failed") == 1


@pytest.mark.asyncio
async def test_cron_sweep(temp_dir):
    db_path = temp_dir / "cron_queue.db"
    report_path = temp_dir / "cron_report.jsonl"

    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(report_path),
        )
    )

    queue = EpisodeQueue(str(db_path))
    await queue.enqueue("Memory 1")
    await queue.enqueue("Memory 2")

    mock_client = AsyncMock()
    mock_client.add_episode = AsyncMock(return_value=None)

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=mock_client):
        with patch.object(queue, "db_path", db_path):
            with patch("janus_graph.pipeline.cron.EpisodeQueue", return_value=queue):
                summary = await run_cron_sweep(settings=settings, batch_size=5)
                assert summary["processed"] == 2
                assert summary["succeeded"] == 2
                assert summary["failed"] == 0

    assert report_path.exists()


@pytest.mark.asyncio
async def test_cron_sweep_reports_total_episodes(temp_dir):
    """The sweep summary must carry the episodes-table total, not a partial slice.

    Three rows go in, and they end in three *different* statuses, so the
    total can only be right if every status is counted:

      - one succeeds        (done)
      - one is left alone   (queued)
      - one is aborted      (aborted, and lands in dead_letter)

    The aborted row is the point of the test: it makes ``get_stats()`` report
    ``dlq=1``, so a report that summed get_stats() would claim 4 episodes for
    a 3-row table. Asserting against a real ``SELECT COUNT(*)`` pins the
    difference instead of trusting any particular status arithmetic.
    """
    db_path = temp_dir / "totep_queue.db"
    report_path = temp_dir / "totep_report.jsonl"

    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(report_path),
        )
    )

    queue = EpisodeQueue(str(db_path))
    done_id = await queue.enqueue("Memory that succeeds")
    await queue.enqueue("Memory that stays queued")
    aborted_id = await queue.enqueue("Memory that is aborted")
    await queue.mark_aborted(aborted_id, "Test error")

    mock_client = AsyncMock()
    mock_client.add_episode = AsyncMock(return_value=None)

    with patch("janus_graph.core.instance.create_graphiti_instance", return_value=mock_client):
        with patch.object(queue, "db_path", db_path):
            with patch("janus_graph.pipeline.cron.EpisodeQueue", return_value=queue):
                summary = await run_cron_sweep(settings=settings, batch_size=1)

    # Ground truth: the table itself, not any aggregation of it.
    with sqlite3.connect(str(db_path)) as conn:
        real_total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    assert "total_episodes" in summary, "sweep summary lost the total_episodes key"
    assert summary["total_episodes"] == real_total
    # Guard the trap explicitly: the DLQ row is not an episode row.
    assert summary["dlq_count"] == 1
    assert summary["total_episodes"] != real_total + summary["dlq_count"]
    # And it must not be confused with the backlog slice it sits next to.
    assert summary["total_episodes"] >= summary["queued_remaining"]
    assert done_id != aborted_id

    # The key must survive into the emitted report, not just the return value.
    assert report_path.exists()
    events = [json.loads(line) for line in report_path.read_text().splitlines() if line.strip()]
    sweeps = [e for e in events if e.get("kind") == "cron_sweep"]
    assert sweeps, "no cron_sweep report was written"
    assert sweeps[-1]["details"]["total_episodes"] == real_total


def test_dream_label_propagation():
    # Simple cluster graph: {A: [B], B: [A], C: [D], D: [C]}
    projection = {
        "A": ["B"],
        "B": ["A"],
        "C": ["D"],
        "D": ["C"],
    }
    communities = bounded_label_propagation(projection)
    assert len(communities) == 2
    # Check that A and B are together, C and D are together
    flat = {node: tuple(c) for c in communities for node in c}
    assert flat["A"] == flat["B"]
    assert flat["C"] == flat["D"]
    assert flat["A"] != flat["C"]


@pytest.mark.asyncio
async def test_dream_consolidation(temp_dir):
    db_path = temp_dir / "dream_queue.db"
    report_path = temp_dir / "dream_report.jsonl"

    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(report_path),
        )
    )

    queue = EpisodeQueue(str(db_path))
    with patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue):
        results = await run_dream_consolidation(settings=settings, force=True)
        assert results["status"] == "completed"
        assert results["phase_1_clustering"] == "DONE"
        assert results["phase_2_deduplication"] == "DONE"
        assert results["phase_3_orphan_pruning"] == "DONE"
        assert "DONE" in results["phase_4_dlq_repair"]
        assert report_path.exists()
