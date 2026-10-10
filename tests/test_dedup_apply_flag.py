"""Phase 2 must be able to WRITE on the nightly cron, without Phase 3.

Context: ``run_dream_consolidation(force=...)`` passed ``force`` straight to
BOTH ``run_phase2_dedup`` (dream.py:196) and ``run_phase3_prune`` (dream.py:230).
The cron loop calls it with no ``force`` at all (cron_loop.py:214), so nightly
Phase 2 had been a permanent dry-run since the split — it planned 2 merges on
the 2026-10-07 snapshot and wrote nothing.

Simply flipping the cron call to ``force=True`` would ALSO have switched Phase 3
on, which deletes orphan nodes (38 on that snapshot). Phase 2 and Phase 3
destroy different things, so Phase 2 gets its own switch: ``DreamConfig.
dedup_apply``, defaulting to False.

These tests pin the three properties that make the switch safe:

  1. ``dedup_apply=True`` really reaches ``apply_plan`` (applied=True, writes).
  2. Default/absent config stays a dry-run — identity, zero writes.
  3. Phase 3 is NOT dragged along by Phase 2's switch.
"""

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from janus_graph.config import DreamConfig
from janus_graph.core.contracts import ReportSettings, Settings
from janus_graph.pipeline.dream import run_dream_consolidation
from janus_graph.pipeline.queue import EpisodeQueue

P2_DONE = {
    "status": "DONE",
    "reason": "",
    "detail": "D",
    "merges": 2,
    "entities_seen": 5788,
    "applied": True,
    "plan": {},
    "counters": {"applied": True, "nodes_deleted": 2},
}

P2_DRY = {
    "status": "DONE",
    "reason": "",
    "detail": "D",
    "merges": 2,
    "entities_seen": 5788,
    "applied": False,
    "plan": {},
    "counters": {
        "applied": False,
        "nodes_deleted": 0,
        "mentions_moved": 0,
        "relations_moved": 0,
        "errors": [],
    },
}

P3_PLANNED = {
    "status": "PLANNED",
    "reason": "",
    "detail": "dry-run: 38 prunable node(s) withheld (force=False)",
    "candidates": 861,
    "prunable": 38,
    "deleted": 0,
    "applied": False,
}


def _settings(tmp_path, name, dream=None):
    """Build Settings whose ``pipeline.dream`` is *dream* when supplied.

    Both ``Settings`` and ``PipelineSettings`` are FROZEN dataclasses, so the
    flag is installed with ``dataclasses.replace`` rather than assignment.

    Note what ``dream`` really is here. ``contracts.PipelineSettings.dream`` is
    ``contracts.DreamSettings`` (a dataclass carrying the UNREACHABLE
    ``dedup_threshold=0.85`` copy), NOT ``config.DreamConfig``. Production
    ``dream.py`` is handed a ``JanusSettings`` whose ``pipeline.dream`` IS a
    ``DreamConfig``. dream.py reads the flag through a defensive getattr chain,
    so both shapes work — this test pins the contracts-side shape because that
    is the one a bare ``Settings()`` actually produces.
    """
    settings = Settings(report=ReportSettings(sinks=("file",), file_path=Path(tmp_path / name)))
    if dream is not None:
        settings = dataclasses.replace(
            settings,
            pipeline=dataclasses.replace(settings.pipeline, dream=dream),
        )
    return settings


def _queue(tmp_path, name):
    return EpisodeQueue(str(tmp_path / name))


@pytest.mark.asyncio
async def test_dedup_apply_flag_makes_phase2_write(tmp_path):
    """Flag ON must reach apply_plan — a plan nobody executes is the bug."""
    settings = _settings(tmp_path, "p2_on.jsonl", dream=DreamConfig(dedup_apply=True))
    queue = _queue(tmp_path, "p2_on.db")
    with (
        patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue),
        patch("janus_graph.pipeline.dedup_apply.run_phase2_dedup", return_value=P2_DONE) as p2,
        patch(
            "janus_graph.pipeline.dream_apply.run_phase1_cluster",
            return_value={"status": "DONE", "communities": 1},
        ),
        patch("janus_graph.pipeline.dream_apply.run_phase3_prune", return_value=P3_PLANNED) as p3,
    ):
        results = await run_dream_consolidation(settings=settings)

    assert p2.call_args.kwargs["force"] is True
    assert results["phase_2_apply"] is True
    assert results["phase_2_merges"] == 2
    assert results["phase_2_detail"]["applied"] is True
    assert results["phase_3_detail"]["applied"] is False


@pytest.mark.asyncio
async def test_dedup_apply_defaults_to_dry_run(tmp_path):
    """Negative control: absent flag must be identity, zero writes.

    Without this the switch would be untestable — a suite that only checks the
    ON path passes just as happily when the default is broken to True.
    """
    settings = _settings(tmp_path, "p2_off.jsonl", dream=DreamConfig())
    queue = _queue(tmp_path, "p2_off.db")
    with (
        patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue),
        patch("janus_graph.pipeline.dedup_apply.run_phase2_dedup", return_value=P2_DRY) as p2,
        patch(
            "janus_graph.pipeline.dream_apply.run_phase1_cluster",
            return_value={"status": "DONE", "communities": 1},
        ),
        patch("janus_graph.pipeline.dream_apply.run_phase3_prune", return_value=P3_PLANNED),
    ):
        results = await run_dream_consolidation(settings=settings)

    assert p2.call_args.kwargs["force"] is False
    assert results["phase_2_apply"] is False
    assert results["phase_2_detail"]["applied"] is False
    assert results["phase_2_detail"]["counters"]["nodes_deleted"] == 0


@pytest.mark.asyncio
async def test_missing_dream_config_is_dry_run_not_crash(tmp_path):
    """A bare contracts.Settings has no pipeline.dream -> False, no exception.

    dream.py receives either JanusSettings or contracts.Settings depending on the
    caller; the getattr chain must not raise on the second one.
    """
    settings = Settings(
        report=ReportSettings(sinks=("file",), file_path=Path(tmp_path / "bare.jsonl"))
    )
    queue = _queue(tmp_path, "bare.db")
    with (
        patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue),
        patch("janus_graph.pipeline.dedup_apply.run_phase2_dedup", return_value=P2_DRY) as p2,
        patch(
            "janus_graph.pipeline.dream_apply.run_phase1_cluster",
            return_value={"status": "DONE", "communities": 1},
        ),
        patch("janus_graph.pipeline.dream_apply.run_phase3_prune", return_value=P3_PLANNED),
    ):
        results = await run_dream_consolidation(settings=settings)

    assert p2.call_args.kwargs["force"] is False
    assert results["phase_2_apply"] is False


@pytest.mark.asyncio
async def test_dedup_apply_does_not_enable_phase3_pruning(tmp_path):
    """THE safety property: P2's switch must not delete the 38 orphans.

    If this ever regresses, enabling nightly dedup silently starts deleting
    graph nodes — the failure mode that made Phase 2 need its own flag.
    """
    settings = _settings(tmp_path, "p2_safe.jsonl", dream=DreamConfig(dedup_apply=True))
    queue = _queue(tmp_path, "p2_safe.db")
    with (
        patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue),
        patch("janus_graph.pipeline.dedup_apply.run_phase2_dedup", return_value=P2_DONE),
        patch(
            "janus_graph.pipeline.dream_apply.run_phase1_cluster",
            return_value={"status": "DONE", "communities": 1},
        ),
        patch("janus_graph.pipeline.dream_apply.run_phase3_prune", return_value=P3_PLANNED) as p3,
    ):
        results = await run_dream_consolidation(settings=settings)

    assert p3.call_args.kwargs["force"] is False
    assert results["phase_3_pruned"] == 0
    assert results["phase_3_detail"]["status"] == "PLANNED"
    assert results["phase_3_detail"]["prunable"] == 38


@pytest.mark.asyncio
async def test_force_still_enables_both_phases(tmp_path):
    """force=True keeps its old meaning: the explicit, whole-run override."""
    settings = _settings(tmp_path, "force.jsonl", dream=DreamConfig())
    queue = _queue(tmp_path, "force.db")
    with (
        patch("janus_graph.pipeline.dream.EpisodeQueue", return_value=queue),
        patch("janus_graph.pipeline.dedup_apply.run_phase2_dedup", return_value=P2_DONE) as p2,
        patch(
            "janus_graph.pipeline.dream_apply.run_phase1_cluster",
            return_value={"status": "DONE", "communities": 1},
        ),
        patch(
            "janus_graph.pipeline.dream_apply.run_phase3_prune",
            return_value={"status": "DONE", "deleted": 38, "applied": True},
        ) as p3,
    ):
        results = await run_dream_consolidation(settings=settings, force=True)

    assert p2.call_args.kwargs["force"] is True
    assert p3.call_args.kwargs["force"] is True
    assert results["phase_3_pruned"] == 38


def test_dedup_apply_flag_default_is_false():
    """A new switch that defaults to True would rewrite history overnight."""
    assert DreamConfig().dedup_apply is False
