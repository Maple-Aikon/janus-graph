"""aiohttp HTTP server for janus-graph daemon (Phase 2 PR scope).

Endpoints (Phase 2):
  GET  /health   — composite health snapshot
                    200 if Falkor alive
                    503 if Falkor down (degraded mode)
  POST /shutdown — graceful shutdown trigger (admin; Phase 2 stub)

Phase 3 endpoints (/episodes, /search, /sweep) intentionally NOT shipped here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import web

from ..config import HTTPSettings
from .circuit_breaker import CircuitState

if TYPE_CHECKING:
    from .lifespan import DaemonContext

logger = logging.getLogger("janus_graph.daemon.http_server")

__version__ = "0.2.0-phase2"


def build_app(ctx: "DaemonContext") -> web.Application:
    """Build aiohttp app bound to daemon context.

    Routes:
      GET  /health   → DaemonContext.health_view
      POST /shutdown → DaemonContext.shutdown_view
    """
    app = web.Application()
    app["ctx"] = ctx
    app.router.add_get("/health", health_handler)
    app.router.add_post("/shutdown", shutdown_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    """Composite health snapshot.

    Returns 200 when Falkor alive, 503 when degraded.
    Body shape (consistent across both — operationally useful):
      {
        "status": "ready" | "degraded",
        "falkor_ok": bool,
        "circuit": {
          "state": "closed" | "open" | "half_open",
          "failure_count": int,
          "failure_threshold": int,
          "reset_timeout_sec": int,
          "retry_at": float | null
        },
        "daemon_version": str,
        "queue_stats": { ... } | null   # Phase 2: always null (queue worker in Phase 3)
      }
    """
    ctx: "DaemonContext" = request.app["ctx"]
    snap = ctx.supervisor.snapshot()

    # Probe Falkor (gated by breaker — bounded retry).
    falkor_ok = await ctx.supervisor.probe()

    payload = {
        "status": "ready" if falkor_ok else "degraded",
        "falkor_ok": falkor_ok,
        "circuit": {
            "state": snap.state.value,
            "failure_count": snap.failure_count,
            "failure_threshold": snap.failure_threshold,
            "reset_timeout_sec": snap.reset_timeout_sec,
            "retry_at": snap.retry_at,
        },
        "daemon_version": __version__,
        "queue_stats": None,  # Phase 3: pipe EpisodeQueue.stats() here
    }

    # Degraded → 503 with same body (operationally useful — clients see state).
    status_code = 200 if falkor_ok else 503
    return web.json_response(payload, status=status_code)


async def shutdown_handler(request: web.Request) -> web.Response:
    """Graceful shutdown trigger (admin).

    Phase 2 stub: respond 202 + ask lifespan to exit.
    Does NOT actually call sys.exit — lifespan loop polls ctx.shutdown_event.
    """
    ctx: "DaemonContext" = request.app["ctx"]
    ctx.request_shutdown()
    return web.json_response(
        {"status": "shutting_down", "daemon_version": __version__},
        status=202,
    )


__all__ = ["build_app", "health_handler", "shutdown_handler", "__version__"]


# Re-export HTTPSettings for convenience.
_re_exports = [HTTPSettings, CircuitState]
