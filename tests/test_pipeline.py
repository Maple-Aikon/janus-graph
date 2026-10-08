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
        # Phase 2 used to be the literal "DONE" with no implementation behind
        # it. It now reports a real status plus the reason, so the assertion is
        # on the STATUS PREFIX, not on the bare word -- otherwise a phase that
        # silently skipped would still pass.
        # The sentinel reason is deliberately distinctive: the REAL function
        # with no database says "FalkorDB unreachable (ConnectionError: ...)",
        # which also contains the word "unreachable". Asserting only on that
        # word would pass whether or not the patch took effect, i.e. the test
        # would not be able to fail. The sentinel proves the patch is live.
        sentinel = "TEST-SENTINEL-phase2-not-a-real-status"
        sentinel3 = "TEST-SENTINEL-phase3-not-a-real-status"
        # Phase 1 needs the same treatment, and this one bit me: the assertion
        # below said "SKIPPED ... with no graph reachable", but nothing made the
        # graph unreachable. So the test passed only while FalkorDB happened to
        # be DOWN. When it came back up mid-session this test failed with
        # 4,187 communities over 5,844 real nodes -- an environment-dependent
        # test, not a regression. Patch the connection so the premise is true
        # by construction instead of by luck.
        with patch(
            # Patch BOTH bindings. dream.py calls dream_apply.run_phase1_cluster,
            # which uses dream_apply's own module-level `_graph` imported at
            # load time (`from .dedup_apply import _graph`). Patching only the
            # source module leaves the name actually called untouched, so phase
            # 1 reached the real FalkorDB and reported DONE with 4,186
            # communities over 5,846 real nodes. This was also ORDER-DEPENDENT:
            # it passed alone and failed when run after test_dream_phases.py.
            # Patching both names removes the dependence on either binding.
            "janus_graph.pipeline.dedup_apply._graph",
            return_value=(None, "ConnectionError: TEST-SENTINEL-no-graph"),
        ), patch(
            "janus_graph.pipeline.dream_apply._graph",
            return_value=(None, "ConnectionError: TEST-SENTINEL-no-graph"),
        ), patch(
            "janus_graph.pipeline.dedup_apply.run_phase2_dedup",
            return_value={
                "status": "SKIPPED",
                "reason": sentinel,
                "merges": 0,
                "applied": False,
            },
        ), patch(
            "janus_graph.pipeline.dream_apply.run_phase3_prune",
            return_value={
                "status": "SKIPPED",
                "reason": sentinel3,
                "candidates": 0,
                "prunable": 0,
                "deleted": 0,
                "applied": False,
                "detail": "no graph connection",
            },
        ):
            results = await run_dream_consolidation(settings=settings, force=True)
        assert results["status"] == "completed"
        # Phase 1 now runs a real partition. With no graph reachable it must
        # say so -- and must NOT reuse the old trick of claiming a gate passed
        # while having clustered nothing.
        assert results["phase_1_clustering"].startswith("SKIPPED")
        assert "NOT RUN" not in results["phase_1_clustering"]
        assert results["phase_2_deduplication"].startswith("SKIPPED")
        assert results["phase_2_deduplication"] != "DONE"
        assert sentinel in results["phase_2_deduplication"]
        assert results["phase_2_merges"] == 0
        # Phase 3 was the bare literal "DONE" with no implementation. Same
        # treatment as phase 2: status plus a distinctive reason.
        assert results["phase_3_orphan_pruning"].startswith("SKIPPED")
        assert results["phase_3_orphan_pruning"] != "DONE"
        assert sentinel3 in results["phase_3_orphan_pruning"]
        assert results["phase_3_pruned"] == 0
        assert "DONE" in results["phase_4_dlq_repair"]
        assert report_path.exists()


@pytest.mark.asyncio
async def test_phase_report_shows_reason_and_detail_together(temp_dir):
    """`detail or reason` dropped reason whenever both existed -- which is
    exactly the case where a SKIPPED phase has a diagnostic worth showing."""
    db_path = temp_dir / "dream_both.db"
    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(temp_dir / "dream_both.jsonl"),
        )
    )
    queue = EpisodeQueue(str(db_path))
    with patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue), patch(
        "janus_graph.pipeline.dedup_apply.run_phase2_dedup",
        return_value={
            "status": "SKIPPED",
            "reason": "REASON-A",
            "detail": "DETAIL-B",
            "merges": 0,
            "applied": False,
        },
    ), patch(
        "janus_graph.pipeline.dream_apply.run_phase3_prune",
        return_value={
            "status": "SKIPPED",
            "reason": "REASON-C",
            "detail": "DETAIL-D",
            "candidates": 0,
            "prunable": 0,
            "deleted": 0,
            "applied": False,
        },
    ):
        results = await run_dream_consolidation(settings=settings, force=True)
    for key, reason, detail in (
        ("phase_2_deduplication", "REASON-A", "DETAIL-B"),
        ("phase_3_orphan_pruning", "REASON-C", "DETAIL-D"),
    ):
        assert reason in results[key], key
        assert detail in results[key], key


@pytest.mark.asyncio
async def test_dream_status_downgrades_when_a_phase_fails(temp_dir):
    """status used to be the unconditional literal "completed", so a run where
    every phase threw still reported success."""
    db_path = temp_dir / "dream_queue_fail.db"
    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(temp_dir / "dream_fail.jsonl"),
        )
    )
    queue = EpisodeQueue(str(db_path))
    with patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue), patch(
        "janus_graph.pipeline.dedup_apply.run_phase2_dedup",
        side_effect=RuntimeError("dedup exploded"),
    ), patch(
        "janus_graph.pipeline.dream_apply.run_phase3_prune",
        side_effect=RuntimeError("prune exploded"),
    ):
        results = await run_dream_consolidation(settings=settings, force=True)
    assert results["status"] != "completed"
    assert results["status"] == "completed_with_failures"
    assert set(results["failed_phases"]) == {
        "phase_2_deduplication",
        "phase_3_orphan_pruning",
    }
    assert "dedup exploded" in results["phase_2_deduplication"]
    assert "prune exploded" in results["phase_3_orphan_pruning"]


@pytest.mark.asyncio
async def test_dream_nodes_before_is_real_not_zero(temp_dir):
    """nodes_before/after were hardcoded 0, i.e. a fabricated measurement."""
    db_path = temp_dir / "dream_nodes.db"
    settings = Settings(
        report=ReportSettings(
            sinks=("file",),
            file_path=Path(temp_dir / "dream_nodes.jsonl"),
        )
    )
    queue = EpisodeQueue(str(db_path))
    with patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue), patch(
        "janus_graph.pipeline.dedup_apply.run_phase2_dedup",
        return_value={
            "status": "DONE",
            "reason": "",
            "merges": 0,
            "applied": False,
            "entities_seen": 42,
            "detail": "merged=0",
        },
    ), patch(
        "janus_graph.pipeline.dream_apply.run_phase3_prune",
        return_value={
            "status": "DONE",
            "reason": "",
            "candidates": 1,
            "prunable": 1,
            "deleted": 2,
            "applied": True,
            "detail": "deleted=2",
        },
    ):
        results = await run_dream_consolidation(settings=settings, force=True)
    assert results["nodes_before"] == 42
    assert results["nodes_after"] == 40
    assert results["phase_3_pruned"] == 2
@pytest.mark.asyncio
async def test_dream_phase1_reports_done_when_the_graph_is_reachable(temp_dir):
    """The other half of the pair the SKIPPED case above pins.

    test_dream_consolidation forces the graph to be unreachable, so this test
    covers the branch that FalkorDB being UP produced: a real partition and a
    DONE status. Two components of three nodes each, so the partition is
    deterministic and hand-checkable: every node is degree 2, so with
    min_degree=3 all six are pinned to their own seed label -> 6 communities
    over 6 nodes, 2 components, largest 3.
    """
    from janus_graph.pipeline.dream_apply import run_phase1_cluster

    snapshot = {
        "entities": [
            {"uuid": "a%d" % i, "name": "a%d" % i, "summary": "", "created_at": None}
            for i in range(3)
        ] + [
            {"uuid": "b%d" % i, "name": "b%d" % i, "summary": "", "created_at": None}
            for i in range(3)
        ],
        "mentions": [],
        "relations": [
            {"uuid": "r0", "src_uuid": "a0", "dst_uuid": "a1", "created_at": None},
            {"uuid": "r1", "src_uuid": "a1", "dst_uuid": "a2", "created_at": None},
            {"uuid": "r2", "src_uuid": "a0", "dst_uuid": "a2", "created_at": None},
            {"uuid": "r3", "src_uuid": "b0", "dst_uuid": "b1", "created_at": None},
            {"uuid": "r4", "src_uuid": "b1", "dst_uuid": "b2", "created_at": None},
            {"uuid": "r5", "src_uuid": "b0", "dst_uuid": "b2", "created_at": None},
        ],
    }

    class _Graph:
        def query(self, _q, *_a, **_kw):
            return []

    with patch(
        # Both names are imported into dream_apply at module load
        # (`from .dedup_apply import _graph, load_graph_snapshot`), so each one
        # must be patched where it is USED. Patching the source module alone
        # silently does nothing, and the test then takes the unreachable branch.
        "janus_graph.pipeline.dream_apply._graph", return_value=(_Graph(), None)
    ), patch(
        # dream_apply does `from .dedup_apply import load_graph_snapshot` at
        # module load, so patching the source module does NOT change the name
        # it actually calls. Patching dedup_apply here did nothing and the test
        # failed with SKIPPED -- the same "patched a name nobody calls" shape
        # that makes a test unable to fail.
        "janus_graph.pipeline.dream_apply.load_graph_snapshot",
        return_value=(snapshot, ""),
    ):
        out = run_phase1_cluster(MagicMock(), "graphiti_memory", min_degree=3)

    assert out["status"] == "DONE"
    assert out["applied"] is False, "phase 1 is a measurement, never a mutation"
    assert out["result"]["nodes"] == 6
    assert out["result"]["components"] == 2
    assert out["result"]["pinned_nodes"] == 6
    assert out["result"]["largest_component"] == 3
    assert out["result"]["oversized"] is False
    # Every node pinned to its own seed label, so the count equals the node
    # count -- and that is why the report says the number is not a measurement.
    assert out["result"]["communities"] == 6
    assert out["result"]["partition_hash"]

    # Same graph, nothing pinned: each triangle unifies into one community.
    with patch(
        # Both names are imported into dream_apply at module load
        # (`from .dedup_apply import _graph, load_graph_snapshot`), so each one
        # must be patched where it is USED. Patching the source module alone
        # silently does nothing, and the test then takes the unreachable branch.
        "janus_graph.pipeline.dream_apply._graph", return_value=(_Graph(), None)
    ), patch(
        # dream_apply does `from .dedup_apply import load_graph_snapshot` at
        # module load, so patching the source module does NOT change the name
        # it actually calls. Patching dedup_apply here did nothing and the test
        # failed with SKIPPED -- the same "patched a name nobody calls" shape
        # that makes a test unable to fail.
        "janus_graph.pipeline.dream_apply.load_graph_snapshot",
        return_value=(snapshot, ""),
    ):
        out0 = run_phase1_cluster(MagicMock(), "graphiti_memory", min_degree=0)
    assert out0["status"] == "DONE"
    assert out0["result"]["communities"] == 2
    assert out0["result"]["largest_community"] == 3
