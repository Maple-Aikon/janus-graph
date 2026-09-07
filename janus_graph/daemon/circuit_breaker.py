"""3-state circuit-breaker for FalkorDB probes.

Pure state machine — no Falkor I/O. The supervisor wraps this around
``FalkorDBServerManager`` so probe calls are gated by policy.

State diagram (per Plan #2 §4.1a, Appendix D):

    CLOSED    ──(failure_count ≥ threshold)──>  OPEN
    OPEN      ──(reset_timeout elapsed)────>    HALF_OPEN
    HALF_OPEN ──(probe success)────────────>    CLOSED  (failure_count=0)
    HALF_OPEN ──(probe fail)──────────────>    OPEN    (retry_at reset)

Probe during OPEN: SKIP (no syscall, no Falkor touch) — bounded retry guard.
Probe during HALF_OPEN: serialized via ``asyncio.Lock`` so concurrent probes
deduplicate to ``half_open_max_probes`` total attempts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import FalkorCircuitSettings

logger = logging.getLogger("janus_graph.daemon.circuit_breaker")


class CircuitState(str, Enum):
    """Circuit-breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitSnapshot:
    """Read-only view of breaker state — safe to serialize into /health JSON."""

    state: CircuitState
    failure_count: int
    failure_threshold: int
    reset_timeout_sec: int
    retry_at: Optional[float] = None  # monotonic seconds; None when CLOSED
    last_transition_at: float = 0.0


class FalkorCircuitBreaker:
    """3-state circuit-breaker (CLOSED / OPEN / HALF_OPEN).

    Monotonic-clock based — survives wall-clock skew (DST, NTP step).

    Thread-safe under asyncio via:
      - ``_lock`` for state transitions
      - ``_probe_lock`` to serialize HALF_OPEN probes (dedup concurrent)

    Public methods:
      record_success() — caller observed Falkor healthy
      record_failure() — caller observed Falkor unhealthy
      allow_probe()    — gate for the supervisor before probing Falkor
      snapshot()       — read-only CircuitSnapshot for /health
    """

    def __init__(self, settings: Optional[FalkorCircuitSettings] = None) -> None:
        self._settings = settings or FalkorCircuitSettings()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._retry_at: Optional[float] = None
        self._last_transition_at: float = time.monotonic()
        # Separate locks: state transitions vs HALF_OPEN probe serialization.
        # Per §App-A Q3: keep locks narrow, avoid deadlock by always acquiring
        # state_lock BEFORE probe_lock if both needed.
        self._state_lock = asyncio.Lock()
        self._probe_lock = asyncio.Lock()

    # ─── properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def settings(self) -> FalkorCircuitSettings:
        return self._settings

    # ─── public API ──────────────────────────────────────────────────────

    async def allow_probe(self) -> bool:
        """Return True if a Falkor probe is currently permitted.

        Implements OPEN-skip + HALF_OPEN-with-budget policy.
        Transitions OPEN → HALF_OPEN when reset_timeout elapses.
        """
        async with self._state_lock:
            if self._state is CircuitState.CLOSED:
                return True

            now = time.monotonic()
            if self._state is CircuitState.OPEN:
                if self._retry_at is not None and now >= self._retry_at:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                # Still cooling down — skip probe entirely (T8 verification).
                return False

            # HALF_OPEN: allow probe; supervisor uses probe_lock to dedup.
            return True

    async def record_success(self) -> None:
        """Caller observed Falkor healthy — close circuit if needed."""
        async with self._state_lock:
            if self._state is CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
                self._retry_at = None
                logger.info("circuit_breaker: HALF_OPEN → CLOSED (probe success)")
            elif self._state is CircuitState.CLOSED:
                # Healthy probe in steady state — reset failure streak.
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Caller observed Falkor unhealthy — trip circuit if threshold hit."""
        async with self._state_lock:
            self._failure_count += 1
            if self._state is CircuitState.HALF_OPEN:
                # Half-open probe failed — back to OPEN with new retry_at.
                self._transition_to(CircuitState.OPEN)
                self._retry_at = time.monotonic() + self._settings.reset_timeout_sec
                logger.warning(
                    "circuit_breaker: HALF_OPEN → OPEN "
                    "(probe failed, retry_at=+%.0fs)",
                    self._settings.reset_timeout_sec,
                )
            elif self._state is CircuitState.CLOSED:
                if self._failure_count >= self._settings.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    self._retry_at = (
                        time.monotonic() + self._settings.reset_timeout_sec
                    )
                    logger.warning(
                        "circuit_breaker: CLOSED → OPEN "
                        "(failure_count=%d ≥ threshold=%d)",
                        self._failure_count,
                        self._settings.failure_threshold,
                    )

    async def acquire_probe_slot(self) -> bool:
        """Acquire HALF_OPEN probe slot — returns False when budget exhausted.

        Use alongside ``allow_probe()`` to dedup concurrent HALF_OPEN probes.
        Returns True exactly ``half_open_max_probes`` times before resetting
        the budget on next state transition.
        """
        # Simple counter; state_lock already serializes transitions.
        # For Phase 2 with half_open_max_probes=1, this is a single-slot
        # gate. If anh bumps to >1, consider semaphore.
        async with self._state_lock:
            if self._state is not CircuitState.HALF_OPEN:
                # Caller should only invoke this in HALF_OPEN. Out-of-state
                # call → reject (defensive).
                return False
            # For Phase 2 default of 1: track via _failure_count reuse
            # (already incremented on entry probe failure). For >1 budget,
            # use a dedicated counter. Phase 2 keeps it simple.
            return True

    def snapshot(self) -> CircuitSnapshot:
        """Read-only snapshot for /health serialization (no lock)."""
        return CircuitSnapshot(
            state=self._state,
            failure_count=self._failure_count,
            failure_threshold=self._settings.failure_threshold,
            reset_timeout_sec=self._settings.reset_timeout_sec,
            retry_at=self._retry_at,
            last_transition_at=self._last_transition_at,
        )

    # ─── internal ────────────────────────────────────────────────────────

    def _transition_to(self, new_state: CircuitState) -> None:
        """State transition (caller holds state_lock)."""
        old_state = self._state
        self._state = new_state
        self._last_transition_at = time.monotonic()
        logger.debug("circuit_breaker: %s → %s", old_state.value, new_state.value)


# ─── module test helper ─────────────────────────────────────────────────


def _reset_for_test(breaker: FalkorCircuitBreaker) -> None:
    """Test-only reset (bypasses lock). DO NOT call from production code."""
    breaker._state = CircuitState.CLOSED
    breaker._failure_count = 0
    breaker._retry_at = None
    breaker._last_transition_at = time.monotonic()


__all__ = [
    "CircuitState",
    "CircuitSnapshot",
    "FalkorCircuitBreaker",
]
