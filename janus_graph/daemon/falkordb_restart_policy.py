"""FalkorDB restart policy — bounded-rate self-healing.

Decoupled from the circuit-breaker state machine (FalkorCircuitBreaker) so
restart cadence stays predictable even when state transitions cycle rapidly.

Per Plan #3 (falkordb-auto-restart-supervisor-20260909):

  - Rate limit: max ``max_per_hour`` restarts per rolling 1-hour window.
  - Cooldown: minimum ``cooldown_sec`` seconds between consecutive restarts.
  - Budget exhausted: transition to ``FATAL_DEGRADED``; emit critical log;
    operator intervention required (no further restart attempts).

Public surface:

    policy = FalkorRestartPolicy(settings)
    policy.on_circuit_opened(snapshot, manager)   # breaker → OPEN
    policy.attempt_restart()                     # does the bounded action
    policy.snapshot()                            # read-only state for /health

The restart is delegated to ``manager.start()`` (FalkorDBServerManager.start
wrapped in asyncio.to_thread by the supervisor). This module is purely the
POLICY layer — it owns the rate-limiter + restart orchestration, NOT the
process mechanics.

Thread-safety: monotonic clock + simple list append. Lock-free because:
  - read+modify is single-event-loop
  - restart budget mutations only happen inside ``attempt_restart()`` which
    is awaited sequentially per FalkorRestartPolicy instance
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Optional, TYPE_CHECKING

from ..config import FalkorRestartPolicySettings
from .circuit_breaker import CircuitSnapshot, _NULL_MANAGER_SENTINEL

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.server import FalkorDBServerManager

logger = logging.getLogger("janus_graph.daemon.falkordb_restart_policy")


class RestartPolicyState(str, Enum):
    """Restart-policy health state (orthogonal to circuit-breaker state)."""

    OK = "ok"                          # within budget, restarts permitted
    COOLING_DOWN = "cooling_down"      # last restart too recent; wait
    BUDGET_EXHAUSTED = "budget_exhausted"  # rate limit hit; need operator
    FATAL_DEGRADED = "fatal_degraded"  # operator-only recovery mode


@dataclass
class RestartPolicySnapshot:
    """Read-only view for /health JSON serialization."""

    state: RestartPolicyState
    restarts_in_window: int
    max_per_hour: int
    seconds_since_last_restart: Optional[float]
    cooldown_sec: int
    last_restart_at: Optional[float] = None


class FalkorRestartPolicy:
    """Bounded-rate restart orchestration for FalkorDBServerManager.

    Owns:
      - rolling 1-hour restart budget (``max_per_hour``)
      - minimum cooldown between consecutive restarts (``cooldown_sec``)
      - callback registration for circuit-breaker CLOSED → OPEN events
    """

    def __init__(
        self,
        settings: Optional[FalkorRestartPolicySettings] = None,
        manager: Optional["FalkorDBServerManager"] = None,
        on_budget_exhausted: Optional[Callable[[RestartPolicySnapshot], None]] = None,
    ) -> None:
        self._settings = settings or FalkorRestartPolicySettings()
        self._manager = manager
        self._on_budget_exhausted = on_budget_exhausted
        # Rolling window of restart timestamps (monotonic seconds).
        self._restart_times: Deque[float] = deque()
        self._last_restart_at: Optional[float] = None
        self._state: RestartPolicyState = RestartPolicyState.OK
        # Phase 2 (Plan #3 §4.2): serialise concurrent restart attempts so a
        # probe storm (cron + MCP + /health) cannot spawn twice. ``threading.Lock``
        # is used because ``attempt_restart()`` is itself sync (manager.start()
        # is sync, wrapped in asyncio.to_thread by the supervisor). Async locks
        # would force us to make the whole pipeline async; not worth it.
        self._restart_in_progress: threading.Lock = threading.Lock()
        # Operator-controlled kill switch — set by lifespan boot check when
        # bin/falkordb.so missing. Pre-empts all gate checks below.
        self._disabled_by_operator: bool = False

    def disable(self, reason: str) -> None:
        """Soft-disable policy (used by lifespan binary-check + future ops kill-switch).

        Idempotent. ``reason`` is logged once at WARNING. After disable(),
        attempt_restart() short-circuits to False without consuming budget
        — so no spam, no state mutation, /health still reports the reason via
        the snapshot ``disabled_reason`` field.
        """
        if self._disabled_by_operator:
            return
        self._disabled_by_operator = True
        logger.warning(
            "restart_policy: DISABLED by operator (reason=%s)", reason
        )

    @property
    def disabled_reason(self) -> Optional[str]:
        """Return the disable reason (None if policy is enabled)."""
        return "operator_disabled" if self._disabled_by_operator else None

    # ─── properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> RestartPolicyState:
        return self._state

    @property
    def settings(self) -> FalkorRestartPolicySettings:
        return self._settings

    @property
    def last_restart_at(self) -> Optional[float]:
        return self._last_restart_at

    # ─── public API ──────────────────────────────────────────────────────

    def snapshot(self) -> RestartPolicySnapshot:
        """Read-only snapshot for /health JSON (no mutation)."""
        self._prune_window()  # keep the count honest at snapshot time
        now = time.monotonic()
        seconds_since = (
            None if self._last_restart_at is None else now - self._last_restart_at
        )
        return RestartPolicySnapshot(
            state=self._state,
            restarts_in_window=len(self._restart_times),
            max_per_hour=self._settings.max_per_hour,
            seconds_since_last_restart=seconds_since,
            cooldown_sec=self._settings.cooldown_sec,
            last_restart_at=self._last_restart_at,
        )

    def register_manager(self, manager: "FalkorDBServerManager") -> None:
        """Late-bind the FalkorDBServerManager (testable)."""
        self._manager = manager

    def can_restart(self) -> bool:
        """Return True iff a restart is currently permitted.

        Pure read — does NOT consume budget. Used by callers that want to
        gate before calling ``attempt_restart()``.
        """
        if self._state is RestartPolicyState.FATAL_DEGRADED:
            return False
        if self._state is RestartPolicyState.BUDGET_EXHAUSTED:
            return False
        self._prune_window()
        if len(self._restart_times) >= self._settings.max_per_hour:
            return False
        # Cooldown check — but do NOT auto-transition state here (state
        # transitions only happen inside attempt_restart, so /health
        # reports reflect the last action, not the last query).
        if self._last_restart_at is not None:
            if (time.monotonic() - self._last_restart_at) < self._settings.cooldown_sec:
                return False
        return True

    def on_circuit_opened(
        self,
        snapshot: CircuitSnapshot,
        manager: Optional["FalkorDBServerManager"] = None,
    ) -> None:
        """Callback fired by circuit-breaker on CLOSED → OPEN.

        Args:
            snapshot: CircuitSnapshot at the time of transition.
            manager: Optional FalkorDBServerManager instance. If passed,
                binds the manager before attempting restart. If omitted
                (sentinel from circuit_breaker), uses the previously-bound
                manager (set via ``register_manager`` from the supervisor).
        """
        logger.warning(
            "restart_policy: circuit CLOSED → OPEN (failures=%d), "
            "attempting bounded restart",
            snapshot.failure_count,
        )
        # Late-bind manager if caller passed one (e.g. supervisor wiring).
        if manager is not None and manager is not _NULL_MANAGER_SENTINEL:
            self.register_manager(manager)
        # Fire-and-forget — caller is in async loop; the daemon supervisor
        # awaits this via attempt_restart(). For pure hook registration
        # without execution, use register_circuit_hook() instead.
        self.attempt_restart()

    def attempt_restart(self) -> bool:
        """Attempt one Falkor restart gated by rate-limiter.

        Returns:
            True iff a restart was issued. False if policy refused.

        Side effects on success:
            - appends monotonic timestamp to rolling window
            - updates ``_last_restart_at`` and ``_state``
            - calls ``manager.start()`` synchronously (caller wraps in
              asyncio.to_thread if running inside async loop)

        On budget exhaustion:
            - transitions to BUDGET_EXHAUSTED → FATAL_DEGRADED
            - fires ``on_budget_exhausted`` callback (one-shot)
            - emits critical log

        Phase 2 (Plan #3 §4.2): concurrent calls are serialised via
        ``self._restart_in_progress`` (asyncio.Lock). The second caller
        short-circuits to False without consuming budget — caller sees
        the lock as "another restart is in flight, do not duplicate".
        """
        if self._disabled_by_operator:
            logger.warning(
                "restart_policy: restart refused (operator_disabled)"
            )
            return False

        if self._state is RestartPolicyState.FATAL_DEGRADED:
            logger.error("restart_policy: restart refused (FATAL_DEGRADED)")
            return False

        # Lock acquisition is non-blocking: if another restart is in flight,
        # skip rather than wait (caller is likely a probe-storm duplicate).
        if self._restart_in_progress.locked():
            logger.debug(
                "restart_policy: restart skipped (concurrent _restart in flight)"
            )
            return False

        with self._restart_in_progress:
            return self._attempt_restart_locked()

    def _attempt_restart_locked(self) -> bool:
        """Inner attempt_restart — assumed to hold ``_restart_in_progress``."""
        self._prune_window()

        # Cooldown gate.
        if self._last_restart_at is not None:
            elapsed = time.monotonic() - self._last_restart_at
            if elapsed < self._settings.cooldown_sec:
                self._state = RestartPolicyState.COOLING_DOWN
                logger.info(
                    "restart_policy: cooling down "
                    "(%.1fs elapsed, need %.0fs)",
                    elapsed, self._settings.cooldown_sec,
                )
                return False

        # Rate-limit gate.
        if len(self._restart_times) >= self._settings.max_per_hour:
            self._enter_fatal_degraded(reason="rate_limit")
            return False

        # Budget available → execute restart.
        if self._manager is None:
            logger.error("restart_policy: no manager bound; cannot restart")
            return False

        try:
            ok = self._manager.start()
        except Exception as e:
            logger.critical(
                "restart_policy: manager.start() raised %s; "
                "entering FATAL_DEGRADED",
                e,
                exc_info=True,
            )
            self._enter_fatal_degraded(reason="manager_exception")
            return False

        now = time.monotonic()
        self._restart_times.append(now)
        self._last_restart_at = now
        self._state = RestartPolicyState.OK
        if ok:
            logger.info(
                "restart_policy: restart succeeded "
                "(restarts_in_window=%d/%d)",
                len(self._restart_times), self._settings.max_per_hour,
            )
        else:
            logger.warning(
                "restart_policy: manager.start() returned False; "
                "FalkorDB may not be fully up",
            )
        return ok

    # ─── internal ────────────────────────────────────────────────────────

    def _prune_window(self) -> None:
        """Drop timestamps older than 1 hour from the rolling window."""
        one_hour_ago = time.monotonic() - 3600.0
        while self._restart_times and self._restart_times[0] < one_hour_ago:
            self._restart_times.popleft()

    def _enter_fatal_degraded(self, reason: str) -> None:
        """Transition to FATAL_DEGRADED; one-shot critical log + callback."""
        if self._state is RestartPolicyState.FATAL_DEGRADED:
            return  # already there; don't re-fire
        self._state = RestartPolicyState.FATAL_DEGRADED
        snap = self.snapshot()
        logger.critical(
            "restart_policy: BUDGET_EXHAUSTED → FATAL_DEGRADED (reason=%s, "
            "restarts_in_window=%d/%d, cooldown_sec=%d). "
            "Operator intervention required.",
            reason,
            snap.restarts_in_window,
            snap.max_per_hour,
            snap.cooldown_sec,
        )
        if self._on_budget_exhausted is not None:
            try:
                self._on_budget_exhausted(snap)
            except Exception:  # pragma: no cover
                logger.exception("restart_policy: on_budget_exhausted callback raised")


__all__ = [
    "FalkorRestartPolicy",
    "RestartPolicyState",
    "RestartPolicySnapshot",
]
