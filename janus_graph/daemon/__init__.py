"""Phase 2 + 3 + 4 daemon package - janus_graph HTTP service + Falkor circuit-breaker.

Scope (per Plan #2 / janus-graph-daemon-architecture-proposal-phase-1-design-20260903):
  Phase 2 (§4.1a, §6, §13 B3-B5): boot flow, GET /health, Falkor supervisor
                                  + circuit-breaker.
  Phase 3 (§6 Phase 3, §Phase 3.1): POST /episodes + GET /search/graph.
  Phase 4 (§4.1b, §6 Phase 4): cron_loop (sweep + dream) gated by
                                daemon.cron_enabled=true.

Public surface (lazy-loaded via PEP 562 to keep import-time light):
  Phase 2: DaemonSupervisor, build_app, run
  Phase 3: EpisodeQueueAdapter, SearchEngine, EmbeddingClient,
           DaemonLockSettings, DaemonSearchGraphSettings
  Phase 4: CronLoop

Lazy imports avoid forcing aiohttp / redis at package import time - unit
tests that only touch circuit_breaker.py can run without those deps.
"""

from __future__ import annotations
__all__ = [
    "DaemonSupervisor",
    "build_app",
    "run",
    # Phase 3
    "EpisodeQueueAdapter",
    "SearchEngine",
    "EmbeddingClient",
    "DaemonLockSettings",
    "DaemonSearchGraphSettings",
    # Phase 4
    "CronLoop",
]


_LAZY_EXPORTS = {
    # Phase 2
    "DaemonSupervisor": "janus_graph.daemon.falkordb_supervisor",
    "build_app": "janus_graph.daemon.http_server",
    "run": "janus_graph.daemon.lifespan",
    # Phase 3
    "EpisodeQueueAdapter": "janus_graph.daemon.episode_queue_adapter",
    "SearchEngine": "janus_graph.daemon.search_engine",
    "EmbeddingClient": "janus_graph.daemon.embedding_client",
    "DaemonLockSettings": "janus_graph.daemon.phase3_settings",
    "DaemonSearchGraphSettings": "janus_graph.daemon.phase3_settings",
    # Phase 4
    "CronLoop": "janus_graph.daemon.cron_loop",
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
