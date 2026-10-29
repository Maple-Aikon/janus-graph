"""Tests for the dream_runs audit trail (T3) and the /health cron payload (T4).

Scope (plan ``dream-mode-nightly-hardening-...-20261023``):

  * T3 — ``dream_runs`` had a schema and 0 writers. After 49 successful
    dream runs the table had 0 rows, so an empty table proved nothing. The
    writer must INSERT at the start of a run and UPDATE at the end, and the
    row must SURVIVE a failed run.
  * T4 — ``/health`` exposed no cron/dream state at all, and
    ``phase_1_communities`` was structurally always 0 because it read a key
    that does not exist at the top level of the phase-1 payload.

Real SQLite in a tmp dir, no network, no production data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from janus_graph.daemon.cron_loop import CronLoop
from janus_graph.daemon.http_server import _cron_payload
from janus_graph.pipeline.dream import _phase1_community_count
from janus_graph.pipeline.queue import EpisodeQueue

# ─── T3: the dream_runs audit trail ─────────────────────────────────────


async def test_record_dream_run_inserts_one_row(temp_queue):
    """The table had a schema but NO writer — 0 rows after 49 runs."""
    run_id = "run-001"
    started = datetime.now(timezone.utc).isoformat()

    await temp_queue.record_dream_run(
        run_id=run_id,
        started_at=started,
        completed_at=None,
        status="running",
    )

    rows = temp_queue.get_dream_runs(limit=10)
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "running"
    assert rows[0]["completed_at"] is None


async def test_update_dream_run_closes_the_row(temp_queue):
    """The end-of-run UPDATE must fill the phase columns."""
    run_id = "run-002"
    started = datetime.now(timezone.utc).isoformat()
    await temp_queue.record_dream_run(run_id=run_id, started_at=started, status="running")

    completed = datetime.now(timezone.utc).isoformat()
    await temp_queue.update_dream_run(
        run_id=run_id,
        completed_at=completed,
        status="completed",
        episodes_evaluated=42,
        phase_1_status="DONE (4650 communities)",
        phase_2_status="SKIPPED (dry-run)",
        phase_3_status="SKIPPED (dry-run)",
        phase_4_status="DONE (24 requeued)",
    )

    rows = temp_queue.get_dream_runs(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["completed_at"] == completed
    assert row["episodes_evaluated"] == 42
    assert row["phase_1_status"] == "DONE (4650 communities)"
    assert row["phase_2_status"] == "SKIPPED (dry-run)"
    assert row["phase_3_status"] == "SKIPPED (dry-run)"
    assert row["phase_4_status"] == "DONE (24 requeued)"


async def test_update_dream_run_does_not_blank_other_columns(temp_queue):
    """A partial update must not overwrite the fields it does not mention."""
    run_id = "run-003"
    started = datetime.now(timezone.utc).isoformat()
    await temp_queue.record_dream_run(
        run_id=run_id,
        started_at=started,
        status="running",
        phase_1_status="DONE",
    )

    await temp_queue.update_dream_run(run_id=run_id, status="completed")

    row = temp_queue.get_dream_runs(limit=1)[0]
    assert row["status"] == "completed"
    assert row["phase_1_status"] == "DONE", "an unset column must keep its value"


async def test_update_dream_run_records_the_error_message(temp_queue):
    """A failed run must record WHY it failed — that is the audit value."""
    run_id = "run-004"
    started = datetime.now(timezone.utc).isoformat()
    await temp_queue.record_dream_run(run_id=run_id, started_at=started, status="running")

    await temp_queue.update_dream_run(
        run_id=run_id,
        status="failed",
        error_message="phase_1_clustering",
    )

    row = temp_queue.get_dream_runs(limit=1)[0]
    assert row["status"] == "failed"
    assert row["error_message"] == "phase_1_clustering"


async def test_a_failing_run_still_leaves_the_row(temp_queue):
    """The row is INSERTed BEFORE the phases run.

    A run that crashes mid-way must still leave evidence that it started —
    that is the whole point of an audit trail.
    """
    run_id = "run-005"
    started = datetime.now(timezone.utc).isoformat()

    # The INSERT happens first...
    await temp_queue.record_dream_run(run_id=run_id, started_at=started, status="running")

    # ...and then a phase blows up. The row must survive.
    await temp_queue.update_dream_run(
        run_id=run_id,
        completed_at=datetime.now(timezone.utc).isoformat(),
        status="completed_with_failures",
        error_message="phase_2_deduplication",
    )

    rows = temp_queue.get_dream_runs(limit=10)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed_with_failures"


async def test_get_dream_runs_returns_newest_first(temp_queue):
    """The audit trail is read as a timeline."""
    base = datetime.now(timezone.utc)
    for i in range(3):
        await temp_queue.record_dream_run(
            run_id=f"run-{i}",
            started_at=(
                base.replace(microsecond=0) + __import__("datetime").timedelta(minutes=i)
            ).isoformat(),
            status="running",
        )

    rows = temp_queue.get_dream_runs(limit=10)

    assert [r["run_id"] for r in rows] == ["run-2", "run-1", "run-0"]


# ─── T4: phase_1_communities was structurally always 0 ──────────────────


def test_phase1_community_count_reads_the_nested_result():
    """``run_phase1_cluster`` nests the count under ``result``.

    The old read was ``phase1.get("communities", 0)`` — a key that does not
    exist at the top level — so the reported count was ALWAYS 0.
    """
    payload = {"status": "DONE", "result": {"communities": 4650, "nodes": 6444}}

    assert _phase1_community_count(payload) == 4650


def test_phase1_community_count_falls_back_to_zero_on_a_missing_result():
    """A malformed payload reports 0 rather than raising inside the phase."""
    assert _phase1_community_count({}) == 0
    assert _phase1_community_count({"result": None}) == 0
    assert _phase1_community_count({"result": {}}) == 0
    assert _phase1_community_count({"result": {"communities": "not an int"}}) == 0


def test_phase1_community_count_accepts_a_top_level_value():
    """A payload that DOES carry the count at the top level still works."""
    assert _phase1_community_count({"communities": 7}) == 7


# ─── T4: the /health cron payload ───────────────────────────────────────


class _FakeCronLoop:
    """Minimal stand-in for ``CronLoop`` — only ``snapshot()`` is used."""

    def __init__(self, snapshot_value):
        self._snapshot = snapshot_value

    def snapshot(self):
        return dict(self._snapshot)


class _FakeCtx:
    def __init__(self, cron_loop):
        self.cron_loop = cron_loop


def test_cron_payload_is_none_when_cron_is_disabled():
    """``ctx.cron_loop`` is Optional — absent when cron is disabled."""
    payload = _cron_payload(_FakeCtx(None))

    assert payload["available"] is False
    assert payload["running"] is False
    assert payload["enabled"] is False


def test_cron_payload_returns_the_snapshot():
    """The snapshot is surfaced so /health shows the nightly state."""
    snapshot = {
        "enabled": True,
        "running": True,
        "tick_count": 12,
        "skipped_circuit_open": 0,
        "last_sweep_at": 123.5,
        "last_sweep_at_iso": "2026-10-23T05:00:00+00:00",
        "last_sweep_stats": {"succeeded": 2},
        "last_dream_at": 456.5,
        "last_dream_at_iso": "2026-10-23T02:30:00+00:00",
        "last_dream_result": {"status": "completed"},
    }
    payload = _cron_payload(_FakeCtx(_FakeCronLoop(snapshot)))

    assert payload["available"] is True
    assert payload["tick_count"] == 12
    assert payload["last_dream_at_iso"] == "2026-10-23T02:30:00+00:00"
    assert payload["last_sweep_stats"] == {"succeeded": 2}


def test_cron_payload_never_raises_on_a_broken_snapshot():
    """A broken cron snapshot must not turn /health into a 500.

    The handler's contract is "200 when Falkor alive, 503 when degraded" —
    a snapshot failure degrades to an error field, never to an exception.
    """

    class _BrokenCronLoop:
        def snapshot(self):
            raise RuntimeError("boom")

    payload = _cron_payload(_FakeCtx(_BrokenCronLoop()))

    assert payload["available"] is False
    assert "boom" in payload["error"]


def test_cron_loop_snapshot_carries_iso_timestamps():
    """The real CronLoop exposes BOTH the monotonic value and the ISO one.

    The monotonic value is only meaningful as a delta within one process
    lifetime; the ISO value is what an operator can actually read, and it is
    comparable across a daemon restart.
    """
    cfg = _FakeCronSettings()
    loop = CronLoop(cfg, circuit_state_provider=lambda: "CLOSED")

    snapshot = loop.snapshot()

    assert "last_sweep_at" in snapshot
    assert "last_sweep_at_iso" in snapshot
    assert "last_dream_at" in snapshot
    assert "last_dream_at_iso" in snapshot
    assert snapshot["last_sweep_at_iso"] is None
    assert snapshot["last_dream_at_iso"] is None


class _FakeCronSettings:
    class daemon:
        cron_enabled = True
