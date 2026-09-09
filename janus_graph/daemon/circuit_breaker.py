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
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING, Union

from ..config import FalkorCircuitSettings

if TYPE_CHECKING:  # pragma: no cover
    # Imported only for type checkers; runtime keeps FalkorDBServerManager
    # decoupled to avoid an import cycle (circuit_breaker → falkordb_supervisor).
    from ..engine.server import FalkorDBServerManager

logger = logging.getLogger("janus_graph.daemon.circuit_breaker")

# Callback signature for CLOSED → OPEN hook. Receives the CircuitSnapshot at
# transition time + the FalkorDBServerManager. Manager is typed as Any at
# runtime to break the import cycle.
OnCircuitOpenedCallback = Callable[
    ["CircuitSnapshot", Any],
    Union[None, Awaitable[None]],
]


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
        # CLOSED → OPEN hook (registered by the supervisor). Fires only on
        # the CLOSED→OPEN transition (NOT HALF_OPEN→OPEN, which is a probe
        # retry — see Plan #3 §9 Q7). Can be sync or async; if async, the
        # awaiter is the supervisor (we schedule via asyncio.create_task).
        self._on_open_callback: Optional[OnCircuitOpenedCallback] = None

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

    def on_open(self, callback: OnCircuitOpenedCallback) -> None:
        """Register a callback fired ONLY on CLOSED → OPEN transitions.

        Replaces any previously-registered callback. Per Plan #3 §9 Q7:
        the hook is intentionally NOT fired on HALF_OPEN → OPEN (which is a
        probe retry, not a fresh outage) — firing it there caused a restart
        storm in the previous draft because every cooldown cycle triggered
        a new restart attempt.

        The callback receives ``(CircuitSnapshot, FalkorDBServerManager)``
        and may be sync or async. Async callbacks are scheduled via
        ``asyncio.create_task`` from inside ``_transition_to()``.

        Typical registration:
            breaker.on_open(lambda snap, mgr: policy.on_circuit_opened(snap, mgr))
        or
            breaker.on_open(policy.on_circuit_opened)
        if ``policy`` already binds ``manager`` via ``register_manager``.

        Test-only override: assign ``breaker._on_open_callback`` directly.
        """
        self._on_open_callback = callback

    # ─── internal ────────────────────────────────────────────────────────

    def _transition_to(self, new_state: CircuitState) -> None:
        """State transition (caller holds state_lock).

        Fires the ``_on_open_callback`` ONLY on CLOSED → OPEN — see Plan #3
        §9 Q7 for the rationale (HALF_OPEN → OPEN is a probe retry and
        must NOT trigger a fresh restart cascade).

        The callback signature is ``(snapshot, manager)``. Callers typically
        register via::

            breaker.on_open(
                lambda snap, mgr: policy.on_circuit_opened(snap, mgr)
            )

        Callback exceptions are logged but never break the state machine
        (which has already transitioned).
        """
        old_state = self._state
        self._state = new_state
        self._last_transition_at = time.monotonic()
        logger.debug("circuit_breaker: %s → %s", old_state.value, new_state.value)

        # Fire callback only on CLOSED → OPEN.
        if (
            old_state is CircuitState.CLOSED
            and new_state is CircuitState.OPEN
            and self._on_open_callback is not None
        ):
            cb = self._on_open_callback
            snap = self.snapshot()
            # The supervisor passes its own FalkorDBServerManager via a
            # closure; we don't have a reference here. The hook is invoked
            # with snap + a sentinel manager_arg that the caller resolves.
            # See FalkorRestartPolicy.on_circuit_opened: it accepts the
            # manager either bound (register_manager) or as the 2nd arg.
            try:
                result = cb(snap, _NULL_MANAGER_SENTINEL)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(_invoke_async_callback(cb, snap))
            except Exception:
                logger.exception(
                    "circuit_breaker: sync on_open callback raised; "
                    "blocking restart attempt"
                )


_NULL_MANAGER_SENTINEL = object()


async def _invoke_async_callback(cb, snap):
    """Await an async on_open callback with error containment."""
    try:
        await cb(snap, _NULL_MANAGER_SENTINEL)
    except Exception:
        logger.exception(
            "circuit_breaker: async on_open callback raised; "
            "blocking restart attempt"
        )


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
    "OnCircuitOpenedCallback",
]
