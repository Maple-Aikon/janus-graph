"""FalkorDB supervisor with circuit-breaker policy.

Wraps (does NOT replace) ``FalkorDBServerManager`` so all Falkor lifecycle
stays in one place. Adds bounded-retry guard via ``FalkorCircuitBreaker``.

Async surface:
  - is_running()  — true iff breaker.allow_probe AND manager.is_running()
  - probe()       — single Falkor probe gated by breaker; updates breaker on result
  - start()       — one-shot start attempt (only if breaker permits)

The supervisor is the single chokepoint that all /health handlers go through,
so the circuit state visible to /health reflects real probe outcomes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import EngineConfig, FalkorCircuitSettings
from ..engine.server import FalkorDBServerManager
from .circuit_breaker import FalkorCircuitBreaker, CircuitSnapshot

logger = logging.getLogger("janus_graph.daemon.falkordb_supervisor")


class DaemonSupervisor:
    """Falkor supervisor with circuit-breaker gating.

    Usage (Phase 2):
        supervisor = DaemonSupervisor(engine_cfg, circuit_cfg)
        await supervisor.probe()  # one-off /health check
        snap = supervisor.snapshot()
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        circuit_config: Optional[FalkorCircuitSettings] = None,
        manager: Optional[FalkorDBServerManager] = None,
    ) -> None:
        self._engine = engine_config or EngineConfig()
        self._circuit_cfg = circuit_config or FalkorCircuitSettings()
        self._manager = manager or FalkorDBServerManager(self._engine)
        self._breaker = FalkorCircuitBreaker(self._circuit_cfg)

    # ─── public ──────────────────────────────────────────────────────────

    async def is_running(self) -> bool:
        """True iff breaker permits probe AND manager reports alive."""
        if not await self._breaker.allow_probe():
            return False
        # Probe via manager (sync) wrapped in to_thread.
        return await asyncio.to_thread(self._manager.is_running)

    async def probe(self) -> bool:
        """Single Falkor probe — gated by circuit-breaker.

        Returns True iff Falkor is alive.
        Updates breaker on result (success → reset, failure → record_failure).
        Does NOT call ``start()`` — that's a separate explicit operation.
        """
        if not await self._breaker.allow_probe():
            # OPEN-skip: no syscall, no Falkor touch (T8 verification).
            logger.debug("falkordb_supervisor: probe skipped (circuit OPEN)")
            return False

        try:
            alive = await asyncio.to_thread(self._manager.is_running)
        except Exception as e:
            # Defensive: any probe exception counts as failure.
            logger.warning("falkordb_supervisor: probe raised %s", e)
            alive = False

        if alive:
            await self._breaker.record_success()
        else:
            await self._breaker.record_failure()
        return alive

    async def start(self) -> bool:
        """Attempt Falkor start (only if breaker permits).

        Returns True iff manager.start() succeeded.
        Does NOT record breaker success — that requires a follow-up probe.
        """
        if not await self._breaker.allow_probe():
            logger.warning("falkordb_supervisor: start skipped (circuit OPEN)")
            return False
        try:
            ok = await asyncio.to_thread(self._manager.start)
            return ok
        except Exception as e:
            logger.error("falkordb_supervisor: start raised %s", e)
            await self._breaker.record_failure()
            return False

    async def stop(self) -> bool:
        """Stop Falkor (no breaker check — operator action)."""
        try:
            return await asyncio.to_thread(self._manager.stop)
        except Exception as e:
            logger.error("falkordb_supervisor: stop raised %s", e)
            return False

    def snapshot(self) -> CircuitSnapshot:
        """Read-only breaker snapshot."""
        return self._breaker.snapshot()

    # ─── introspection ───────────────────────────────────────────────────

    @property
    def breaker(self) -> FalkorCircuitBreaker:
        """Direct breaker access (for tests; do not mutate)."""
        return self._breaker

    @property
    def manager(self) -> FalkorDBServerManager:
        """Direct manager access (for tests; do not mutate)."""
        return self._manager


__all__ = ["DaemonSupervisor"]
