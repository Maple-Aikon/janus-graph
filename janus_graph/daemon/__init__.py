"""Phase 2 daemon package — janus_graph HTTP service + Falkor circuit-breaker.

Scope (per Plan #2 / janus-graph-daemon-architecture-proposal-phase-1-design-20260903):
  §4.1a boot flow: parse config → init supervisor → warm graphiti → start HTTP → ready
  §4.1b periodic maintenance task (Phase 4, NOT shipped here): B3 hard rule — no periodic-task import.
  §6 Phase 2 verify gates: T1 (happy), T6 (Falkor down → 503), T8 (3 kills → circuit_open).

Public surface (lazy-loaded via PEP 562 to keep import-time light):
  DaemonSupervisor   — Falkor supervisor with circuit-breaker policy
  build_app          — aiohttp app factory (handlers: GET /health)
  run                — entrypoint (parses config, runs lifespan)

Lazy imports avoid forcing aiohttp at package import time — unit tests
that only touch circuit_breaker.py can run without aiohttp installed.
"""

from __future__ import annotations

__all__ = [
    "DaemonSupervisor",
    "build_app",
    "run",
]


_LAZY_EXPORTS = {
    "DaemonSupervisor": "janus_graph.daemon.falkordb_supervisor",
    "build_app": "janus_graph.daemon.http_server",
    "run": "janus_graph.daemon.lifespan",
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    """Lazy import for public surface (pulls submodules on first access)."""
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent access
        return attr
    raise AttributeError(f"module 'janus_graph.daemon' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(__all__))

