"""Tests for FalkorCircuitBreaker (3-state machine).

Per Plan #2 §6 verification gates T8/T9 + Appendix D:
  - CLOSED → OPEN at failure_threshold (default 3)
  - OPEN skip-probe for reset_timeout_sec (default 60s, mocked in tests)
  - OPEN → HALF_OPEN after reset_timeout
  - HALF_OPEN success → CLOSED + failure_count reset
  - HALF_OPEN failure → OPEN + retry_at reset
  - Concurrent probes during HALF_OPEN serialized via Lock

All tests mock time.monotonic via monkeypatch so we don't sleep 60s in CI.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from janus_graph.config import FalkorCircuitSettings
from janus_graph.daemon.circuit_breaker import (
    CircuitState,
    FalkorCircuitBreaker,
)


@pytest.fixture
def fast_breaker():
    """Breaker with tiny thresholds so tests don't sleep."""
    settings = FalkorCircuitSettings(
        failure_threshold=3,
        reset_timeout_sec=2,  # 2s for OPEN→HALF_OPEN in tests
        half_open_max_probes=1,
    )
    return FalkorCircuitBreaker(settings)


@pytest.fixture
def mock_clock(monkeypatch):
    """Mock time.monotonic so we can advance the clock without sleep."""
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    def _advance(seconds: float) -> None:
        fake_now[0] += seconds

    monkeypatch.setattr(time, "monotonic", _now)
    return _advance


# ─── T8 group: state transitions ──────────────────────────────────────────


async def test_closed_to_open_at_threshold(fast_breaker):
    """CLOSED → OPEN when failure_count ≥ failure_threshold (default 3)."""
    assert fast_breaker.state is CircuitState.CLOSED

    await fast_breaker.record_failure()  # 1
    assert fast_breaker.state is CircuitState.CLOSED
    await fast_breaker.record_failure()  # 2
    assert fast_breaker.state is CircuitState.CLOSED
    await fast_breaker.record_failure()  # 3 → OPEN
    assert fast_breaker.state is CircuitState.OPEN
    assert fast_breaker.failure_count == 3


async def test_closed_success_resets_failure_count(fast_breaker):
    """Successes in CLOSED state reset the failure streak."""
    await fast_breaker.record_failure()
    await fast_breaker.record_failure()
    assert fast_breaker.failure_count == 2

    await fast_breaker.record_success()
    assert fast_breaker.failure_count == 0
    assert fast_breaker.state is CircuitState.CLOSED


# ─── T8 group: OPEN-skip probe behavior ───────────────────────────────────


async def test_open_state_skips_probe(fast_breaker, mock_clock):
    """During OPEN cooldown, allow_probe returns False (no syscall)."""
    # Trip the breaker.
    for _ in range(3):
        await fast_breaker.record_failure()
    assert fast_breaker.state is CircuitState.OPEN

    # Probe immediately — should be skipped.
    assert await fast_breaker.allow_probe() is False

    # Advance 1s (less than reset_timeout_sec=2).
    mock_clock(1.0)
    assert await fast_breaker.allow_probe() is False

    # Advance another 1.5s (total 2.5s ≥ reset_timeout_sec=2) → HALF_OPEN.
    mock_clock(1.5)
    assert await fast_breaker.allow_probe() is True
    assert fast_breaker.state is CircuitState.HALF_OPEN


async def test_open_to_half_open_after_reset_timeout(fast_breaker, mock_clock):
    """OPEN transitions to HALF_OPEN exactly when reset_timeout elapses."""
    for _ in range(3):
        await fast_breaker.record_failure()
    assert fast_breaker.state is CircuitState.OPEN
    snap_before = fast_breaker.snapshot()
    assert snap_before.retry_at is not None

    # Just before reset_timeout → still OPEN.
    mock_clock(fast_breaker.settings.reset_timeout_sec - 0.1)
    assert fast_breaker.state is CircuitState.OPEN

    # At/after reset_timeout → HALF_OPEN on first allow_probe.
    mock_clock(0.2)  # crosses the boundary
    assert await fast_breaker.allow_probe() is True
    assert fast_breaker.state is CircuitState.HALF_OPEN


# ─── T9 group: HALF_OPEN outcome handling ──────────────────────────────────


async def test_half_open_success_closes_circuit(fast_breaker, mock_clock):
    """HALF_OPEN + success → CLOSED + failure_count reset."""
    for _ in range(3):
        await fast_breaker.record_failure()
    mock_clock(fast_breaker.settings.reset_timeout_sec + 0.1)
    assert await fast_breaker.allow_probe() is True
    assert fast_breaker.state is CircuitState.HALF_OPEN

    await fast_breaker.record_success()
    assert fast_breaker.state is CircuitState.CLOSED
    assert fast_breaker.failure_count == 0
    snap = fast_breaker.snapshot()
    assert snap.retry_at is None


async def test_half_open_failure_reopens_with_new_retry(fast_breaker, mock_clock):
    """HALF_OPEN + failure → OPEN + retry_at reset to now+reset_timeout."""
    # Trip → wait → HALF_OPEN.
    for _ in range(3):
        await fast_breaker.record_failure()
    mock_clock(fast_breaker.settings.reset_timeout_sec + 0.1)
    assert await fast_breaker.allow_probe() is True

    # Record failure during HALF_OPEN.
    await fast_breaker.record_failure()
    assert fast_breaker.state is CircuitState.OPEN

    snap = fast_breaker.snapshot()
    assert snap.retry_at is not None
    # New retry_at should be after the previous transition (which was reset_timeout+0.1 ago).
    # We don't pin the exact value, but it must be > now (still OPEN).
    assert snap.retry_at > time.monotonic()


# ─── T9 group: concurrent probe dedup ──────────────────────────────────────


async def test_concurrent_half_open_probes_allow_probe_is_atomic(fast_breaker, mock_clock):
    """Concurrent allow_probe() calls during HALF_OPEN don't race state.

    Multiple coros hit allow_probe simultaneously; only one observes
    the OPEN→HALF_OPEN transition (others see HALF_OPEN).
    """
    for _ in range(3):
        await fast_breaker.record_failure()
    mock_clock(fast_breaker.settings.reset_timeout_sec + 0.1)

    # 10 concurrent allow_probe calls.
    results = await asyncio.gather(*[fast_breaker.allow_probe() for _ in range(10)])
    # All should return True once reset elapsed (state_lock serializes).
    assert all(results)
    assert fast_breaker.state is CircuitState.HALF_OPEN


async def test_concurrent_record_failure_serialized(fast_breaker):
    """Concurrent failures don't race past the threshold (state_lock)."""
    # 10 concurrent failures with threshold=3.
    await asyncio.gather(*[fast_breaker.record_failure() for _ in range(10)])
    assert fast_breaker.state is CircuitState.OPEN
    # failure_count must equal exactly 10 (no lost increments).
    assert fast_breaker.failure_count == 10


# ─── Snapshot tests ───────────────────────────────────────────────────────


async def test_snapshot_shape(fast_breaker, mock_clock):
    """Snapshot has all fields needed by /health JSON."""
    snap = fast_breaker.snapshot()
    assert snap.state is CircuitState.CLOSED
    assert snap.failure_count == 0
    assert snap.failure_threshold == 3
    assert snap.reset_timeout_sec == 2
    assert snap.retry_at is None
    # last_transition_at captured at __init__ (before mock_clock took effect).
    # Cannot compare to time.monotonic() post-mock (returns 0.0). Just assert
    # the field is present and is a float.
    assert isinstance(snap.last_transition_at, float)


async def test_snapshot_after_open_includes_retry_at(fast_breaker, mock_clock):
    """After tripping OPEN, snapshot.retry_at is set to a future timestamp."""
    for _ in range(3):
        await fast_breaker.record_failure()
    snap = fast_breaker.snapshot()
    assert snap.state is CircuitState.OPEN
    assert snap.retry_at is not None
    assert snap.retry_at > time.monotonic()
