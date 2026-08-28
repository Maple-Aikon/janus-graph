"""Tests for Pipeline Worker, Cron Sweeper, and Dream Mode Consolidation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import pytest

from janus_graph.pipeline.queue import EpisodeQueue, EpisodeRecord
from janus_graph.pipeline.worker import EpisodeWorker
from janus_graph.pipeline.cron import run_cron_sweep
from janus_graph.pipeline.dream import (
    bounded_label_propagation,
    run_dream_consolidation,
)
from janus_graph.core.contracts import Settings, PipelineSettings, ReportSettings


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
