"""aiohttp HTTP server for janus-graph daemon (Phase 2 + Phase 3 PR scope).

Endpoints:
  GET  /health        - composite health snapshot
                        200 if Falkor alive
                        503 if Falkor down (degraded mode)
  POST /shutdown      - graceful shutdown trigger (admin)
  POST /episodes      - enqueue episode with dedup + advisory lock (Phase 3)
                        202 + episode_id on success
                        409 DUP / LOCK_MODE_MCP_ONLY / LOCKED on policy hit
                        503 FALKOR_DOWN if backend unreachable
                        422 on malformed body
  GET  /search/graph  - BFS + cosine + MMR structural search (Phase 3.1)
                        200 with {results, count, elapsed_ms, degraded, warnings}
                        400 INVALID_SEED / INVALID_HOPS / INVALID_LIMIT
                        404 SEED_NOT_FOUND
                        422 MIN_COSINE_OUT_OF_RANGE
                        503 FALKOR_DOWN
                        504 TRAVERSAL_TIMEOUT

Phase 3 invariants preserved from Phase 2:
  - 503 body shape (T9): same {status, falkor_ok, circuit, daemon_version,
    queue_stats} envelope so existing clients don't break.
  - cron_loop count == 0 in daemon/__init__.py (B3 hard rule).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from aiohttp import web

from ..config import HTTPSettings
from .circuit_breaker import CircuitState
from .episode_queue_adapter import (
    EpisodeDuplicateError,
    LockModeError,
)
from .search_engine import (
    SearchBackendError,
    SearchValidationError,
)

if TYPE_CHECKING:
    from .lifespan import DaemonContext

logger = logging.getLogger("janus_graph.daemon.http_server")

__version__ = "0.3.0-phase3"


def build_app(ctx: "DaemonContext") -> web.Application:
    """Build aiohttp app bound to daemon context.

    Routes:
      GET  /health         → health_handler
      POST /shutdown       → shutdown_handler
      POST /episodes       → episodes_handler   (Phase 3)
      GET  /search/graph   → search_graph_handler (Phase 3.1)
    """
    app = web.Application()
    app["ctx"] = ctx
    app.router.add_get("/health", health_handler)
    app.router.add_post("/shutdown", shutdown_handler)
    app.router.add_post("/episodes", episodes_handler)
    app.router.add_get("/search/graph", search_graph_handler)
    return app


# ─── /health (Phase 2 — unchanged shape) ───────────────────────────────


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
        "queue_stats": { ... } | null
      }
    """
    ctx: "DaemonContext" = request.app["ctx"]
    snap = ctx.supervisor.snapshot()

    # Probe Falkor (gated by breaker — bounded retry).
    falkor_ok = await ctx.supervisor.probe()

    # Phase 3: surface EpisodeQueueAdapter stats if wired.
    queue_stats = None
    if getattr(ctx, "queue_adapter", None) is not None:
        try:
            queue_stats = ctx.queue_adapter.stats()
        except Exception as e:  # pragma: no cover — defensive
            logger.debug("queue stats error: %s", e)
            queue_stats = {"error": str(e)}

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
        "queue_stats": queue_stats,
    }

    # Degraded → 503 with same body (operationally useful — clients see state).
    status_code = 200 if falkor_ok else 503
    return web.json_response(payload, status=status_code)


# ─── /shutdown (Phase 2 — unchanged) ───────────────────────────────────


async def shutdown_handler(request: web.Request) -> web.Response:
    """Graceful shutdown trigger (admin)."""
    ctx: "DaemonContext" = request.app["ctx"]
    ctx.request_shutdown()
    return web.json_response(
        {"status": "shutting_down", "daemon_version": __version__},
        status=202,
    )


# ─── /episodes (Phase 3) ──────────────────────────────────────────────


async def episodes_handler(request: web.Request) -> web.Response:
    """POST /episodes — enqueue with dedup + advisory lock.

    Request body:
      {
        "content": str,                  # required, non-empty
        "name": str,                     # optional, default = "ep_<uuid8>"
        "group_id": str,                 # optional, default = "graphiti_memory"
        "source_description": str,       # optional, default = "daemon_http"
        "query": str                     # optional, echo for client correlation
      }

    Responses:
      202 + {episode_id, claimed_by, deduplicated:false, daemon_version}
      409 + {code: "DUPLICATE_EPISODE", existing_episode_id}     on hash dedup hit
      409 + {code: "LOCK_MODE_MCP_ONLY", message}                  if mode=mcp_only
      422 + {code: "INVALID_BODY", message}                       on schema violation
      503 + {code: "FALKOR_DOWN"}                                 if backend unreachable
    """
    ctx: "DaemonContext" = request.app["ctx"]
    queue_adapter = getattr(ctx, "queue_adapter", None)
    if queue_adapter is None:
        return web.json_response(
            {"code": "NOT_READY", "message": "queue adapter not initialized"},
            status=503,
        )

    # Body parse (aiohttp: await .json() raises on bad JSON → 400).
    try:
        body = await request.json()
    except Exception as e:
        return web.json_response(
            {"code": "INVALID_BODY", "message": f"malformed JSON: {e}"},
            status=422,
        )

    if not isinstance(body, dict):
        return web.json_response(
            {"code": "INVALID_BODY", "message": "body must be a JSON object"},
            status=422,
        )

    content = body.get("content")
    if not isinstance(content, str) or not content.strip():
        return web.json_response(
            {"code": "INVALID_BODY", "message": "content must be a non-empty string"},
            status=422,
        )

    name = body.get("name")
    if name is not None and not isinstance(name, str):
        return web.json_response(
            {"code": "INVALID_BODY", "message": "name must be a string"},
            status=422,
        )

    group_id = body.get("group_id", "graphiti_memory")
    if not isinstance(group_id, str) or not group_id:
        return web.json_response(
            {"code": "INVALID_BODY", "message": "group_id must be a non-empty string"},
            status=422,
        )

    source_description = body.get("source_description", "daemon_http")
    if not isinstance(source_description, str):
        return web.json_response(
            {"code": "INVALID_BODY", "message": "source_description must be a string"},
            status=422,
        )

    try:
        result = await queue_adapter.enqueue_guarded(
            content=content,
            group_id=group_id,
            name=name,
            source_description=source_description,
        )
    except LockModeError as e:
        return web.json_response(
            {"code": "LOCK_MODE_MCP_ONLY", "message": str(e)},
            status=409,
        )
    except EpisodeDuplicateError as e:
        return web.json_response(
            {
                "code": "DUPLICATE_EPISODE",
                "existing_episode_id": e.existing_episode_id,
            },
            status=409,
        )
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("/episodes enqueue failed")
        return web.json_response(
            {"code": "INTERNAL", "message": f"enqueue failed: {e}"},
            status=503,
        )

    return web.json_response(
        {
            "episode_id": result.episode_id,
            "claimed_by": result.claimed_by,
            "deduplicated": result.deduplicated,
            "daemon_version": __version__,
        },
        status=202,
    )


# ─── /search/graph (Phase 3.1) ────────────────────────────────────────


async def search_graph_handler(request: web.Request) -> web.Response:
    """GET /search/graph — BFS + cosine + MMR structural search.

    Query params (all JSON-encoded in body — we accept POST-style body too
    for ergonomics with hooks that POST JSON):
      seed_entities: list[str]   required, 1-5 entries
      max_hops:      int         required, 1-3
      min_cosine:    float       optional
      mmr_lambda:    float       optional
      limit:         int         optional
      edge_types:    list[str]   optional
      include_episodes: bool     optional
      query:         str         optional (defaults to first seed)

    Responses:
      200 + {results, count, elapsed_ms, degraded, warnings, daemon_version}
      400 + {code, message}      on validation error
      404 + {code: SEED_NOT_FOUND}
      422 + {code: MIN_COSINE_OUT_OF_RANGE}
      503 + {code: FALKOR_DOWN}
      504 + {code: TRAVERSAL_TIMEOUT}
    """
    ctx: "DaemonContext" = request.app["ctx"]
    search_engine = getattr(ctx, "search_engine", None)
    if search_engine is None:
        return web.json_response(
            {"code": "NOT_READY", "message": "search engine not initialized"},
            status=503,
        )

    settings = ctx.settings.daemon.search_graph
    if not settings.enabled:
        return web.json_response(
            {"code": "DISABLED", "message": "search_graph disabled in config"},
            status=503,
        )

    # Accept both GET query string and POST body. For GET (the documented
    # verb in the plan), parse query string into the dict shape.
    if request.method == "GET":
        payload: Dict[str, Any] = {}
        for key in (
            "seed_entities",
            "max_hops",
            "min_cosine",
            "mmr_lambda",
            "limit",
            "edge_types",
            "include_episodes",
            "query",
        ):
            raw = request.query.get(key)
            if raw is None:
                continue
            if key == "seed_entities" or key == "edge_types":
                # Comma-separated for ergonomics (?seed_entities=a,b,c).
                payload[key] = [s.strip() for s in raw.split(",") if s.strip()]
            elif key in ("max_hops", "limit"):
                try:
                    payload[key] = int(raw)
                except ValueError:
                    return web.json_response(
                        {"code": "INVALID_BODY", "message": f"{key} must be int"},
                        status=400,
                    )
            elif key == "include_episodes":
                payload[key] = raw.lower() in ("1", "true", "yes")
            else:
                payload[key] = raw
    else:
        # POST variant for hooks / clients that prefer JSON body.
        try:
            payload = await request.json()
        except Exception as e:
            return web.json_response(
                {"code": "INVALID_BODY", "message": f"malformed JSON: {e}"},
                status=422,
            )

    # Set default query to first seed if not provided (used by cosine step).
    if "query" not in payload:
        seeds = payload.get("seed_entities") or []
        if isinstance(seeds, list) and seeds:
            payload["query"] = seeds[0]

    try:
        result = await search_engine.search(payload)
    except SearchValidationError as e:
        status = 422 if e.code == "MIN_COSINE_OUT_OF_RANGE" else 400
        return web.json_response(
            {"code": e.code, "message": e.message}, status=status
        )
    except SearchBackendError as e:
        if e.code == "SEED_NOT_FOUND":
            return web.json_response({"code": e.code, "message": e.message}, status=404)
        if e.code == "TRAVERSAL_TIMEOUT":
            return web.json_response({"code": e.code, "message": e.message}, status=504)
        # FALKOR_DOWN + default → 503
        return web.json_response({"code": e.code, "message": e.message}, status=503)
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("/search/graph pipeline failed")
        return web.json_response(
            {"code": "INTERNAL", "message": f"search failed: {e}"},
            status=503,
        )

    payload_out = result.to_dict()
    payload_out["daemon_version"] = __version__
    return web.json_response(payload_out, status=200)
