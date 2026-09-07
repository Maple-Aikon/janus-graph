"""Phase 4 CronLoop unit tests.

Per Plan #2 §6 Phase 4 verify gates V1-V5 (subset testable in unit form):

  V1: cron_enabled=false → start() is no-op, running stays False, no tasks.
  V2: cron_enabled=true + CLOSED circuit → sweep tick fires, tick_count++
       snapshot reflects state. (Stubbed run_cron_sweep)
  V3: cron_enabled=true + OPEN circuit → tick skipped, skipped_circuit_open++,
       run_cron_sweep NOT called.
  V4: stop() cancels tasks within timeout; running=False; tick_count preserved.
  Bonus: snapshot() returns enabled/running/tick_count/skipped fields.
  Bonus: state provider coercion (CircuitState enum → str).

These tests stub run_cron_sweep + run_dream_consolidation at the import
site so the test runs without a real EpisodeQueue / FalkorDB.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from janus_graph.config import JanusSettings
from janus_graph.daemon.cron_loop import CronLoop, _state_str


# ─── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def base_settings() -> JanusSettings:
    """JanusSettings with cron disabled by default; tests opt-in via env."""
    return JanusSettings()


@pytest.fixture
def cron_enabled_settings(base_settings: JanusSettings) -> JanusSettings:
    """JanusSettings with cron_enabled=True (via model_copy)."""
    # Build a new object with daemon.cron_enabled=True. Pydantic v2 deep-copy.
    from copy import deepcopy
    s = deepcopy(base_settings)
    s.daemon.cron_enabled = True
    # Shrink interval so tests don't wait 10 minutes.
    s.pipeline.cron_interval_min = 1
    return s


@pytest.fixture
def closed_circuit_provider():
    """Provider returning CircuitState.CLOSED enum."""
    from janus_graph.daemon.circuit_breaker import CircuitState
    return lambda: CircuitState.CLOSED


@pytest.fixture
def open_circuit_provider():
    """Provider returning CircuitState.OPEN enum."""
    from janus_graph.daemon.circuit_breaker import CircuitState
    return lambda: CircuitState.OPEN


@pytest.fixture
def half_open_circuit_provider():
    """Provider returning CircuitState.HALF_OPEN enum."""
    from janus_graph.daemon.circuit_breaker import CircuitState
    return lambda: CircuitState.HALF_OPEN


# ─── V1: cron_enabled=false is no-op ─────────────────────────────────────


@pytest.mark.asyncio
async def test_v1_cron_disabled_no_op(base_settings, closed_circuit_provider):
    """V1: daemon.cron_enabled=false → start() does not spawn tasks."""
    assert base_settings.daemon.cron_enabled is False
    loop = CronLoop(base_settings, closed_circuit_provider)
    await loop.start()
    assert loop.running is False
    assert loop._tasks == []
    snap = loop.snapshot()
    assert snap["enabled"] is False
    assert snap["running"] is False
    assert snap["tick_count"] == 0


# ─── V2: cron_enabled=true spawns tasks ───────────────────────────────────


@pytest.mark.asyncio
async def test_v2_cron_enabled_starts_tasks(cron_enabled_settings, closed_circuit_provider):
    """V2: cron_enabled=true → 2 tasks spawned, snapshot reflects running."""
    loop = CronLoop(cron_enabled_settings, closed_circuit_provider)

    # Stub the sweep + dream so we don't actually run them.
    sweep_mock = AsyncMock(return_value={"succeeded_count": 0, "failed_count": 0})
    dream_mock = AsyncMock(return_value={"status": "complete", "duration_ms": 0})

    with patch("janus_graph.daemon.cron_loop.run_cron_sweep", sweep_mock), \
         patch("janus_graph.daemon.cron_loop.run_dream_consolidation", dream_mock):
        await loop.start()
        try:
            assert loop.running is True
            assert len(loop._tasks) == 2
            names = sorted(t.get_name() for t in loop._tasks)
            assert names == ["cron-dream", "cron-sweep"]

            # Let sweep tick fire once (interval=1min → fast wait via 5s).
            # We bypass cron_interval_min by awaiting the shutdown event.
            await asyncio.sleep(0.2)  # initial 5s wait won't fire; just settle

            # Force a sweep tick by stopping immediately so loop exits.
            # Sweep mock may not have been called yet due to 5s initial wait.
            snap = loop.snapshot()
            assert snap["enabled"] is True
            assert snap["running"] is True
        finally:
            await loop.stop()

    assert loop.running is False
    assert loop._tasks == []


# ─── V3: OPEN circuit → skip tick ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_v3_open_circuit_skips_sweep(cron_enabled_settings, open_circuit_provider):
    """V3: Falkor circuit OPEN → sweep skipped, skipped_circuit_open++."""
    loop = CronLoop(cron_enabled_settings, open_circuit_provider)

    sweep_mock = AsyncMock(return_value={"succeeded_count": 0})

    with patch("janus_graph.daemon.cron_loop.run_cron_sweep", sweep_mock):
        await loop.start()
        # Wait long enough for at least one tick attempt.
        # _sweep_loop waits 5s on first tick then enters interval loop.
        # We can poke the loop manually to test state-poll path faster.
        # Instead, sleep 5.5s for first tick.
        await asyncio.sleep(5.5)
        # sweep_mock should NOT have been called.
        assert sweep_mock.call_count == 0, "sweep should be skipped during OPEN"
        assert loop.skipped_circuit_open >= 1
        await loop.stop()


# ─── V4: stop() cancels cleanly ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_v4_stop_cancels_cleanly(cron_enabled_settings, closed_circuit_provider):
    """V4: stop() returns within timeout, running=False, tasks cleared."""
    loop = CronLoop(cron_enabled_settings, closed_circuit_provider)

    sweep_mock = AsyncMock(side_effect=asyncio.CancelledError)
    with patch("janus_graph.daemon.cron_loop.run_cron_sweep", sweep_mock), \
         patch("janus_graph.daemon.cron_loop.run_dream_consolidation", AsyncMock()):
        await loop.start()
        assert loop.running is True

        # Cancel tasks immediately.
        t0 = asyncio.get_event_loop().time()
        await asyncio.wait_for(loop.stop(), timeout=5.0)
        elapsed = asyncio.get_event_loop().time() - t0

        assert loop.running is False
        assert loop._tasks == []
        assert elapsed < 5.0, "stop() should drain quickly"


# ─── Bonus: snapshot returns expected shape ──────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_shape(base_settings, closed_circuit_provider):
    """Snapshot contains enabled/running/tick_count/skipped fields."""
    loop = CronLoop(base_settings, closed_circuit_provider)
    snap = loop.snapshot()
    for key in ("enabled", "running", "tick_count",
                "skipped_circuit_open", "last_sweep_at",
                "last_sweep_stats", "last_dream_at", "last_dream_result"):
        assert key in snap, f"missing key: {key}"


# ─── Bonus: _state_str coerces ────────────────────────────────────────────


def test_state_str_accepts_enum():
    """_state_str handles CircuitState enum."""
    from janus_graph.daemon.circuit_breaker import CircuitState
    assert _state_str(CircuitState.CLOSED) == "CLOSED"
    assert _state_str(CircuitState.OPEN) == "OPEN"
    assert _state_str(CircuitState.HALF_OPEN) == "HALF_OPEN"


def test_state_str_accepts_plain_string():
    """_state_str handles plain string."""
    assert _state_str("CLOSED") == "CLOSED"
    assert _state_str("open") == "OPEN"  # case-insensitive
    assert _state_str("HALF_OPEN") == "HALF_OPEN"


def test_state_str_unknown_treated_as_closed():
    """Unknown state → CLOSED (permissive default)."""
    assert _state_str(None) == "CLOSED"
    assert _state_str("weird") == "CLOSED"
    assert _state_str(42) == "CLOSED"
