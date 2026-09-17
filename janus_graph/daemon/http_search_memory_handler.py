"""PR 2 / v0.5.0 — HTTP ``/search/memory`` handler.

Single endpoint that delegates to the engine facade
(``janus_graph.engine.search_memory.search_memory``). The MCP transport
(stdio) and this HTTP transport both call the SAME facade function, so
recall semantics are guaranteed identical (Plan §SoT Contract).

Difference vs MCP: HTTP applies:

  - F5 rate limit (60 req/min/IP, token bucket, in-process state)
  - Input validation (query length, limit range, single-string reject
    for diacritic policy — engine owns the dual-string contract)
  - HTTP envelope mapping (status codes, elapsed_ms)
  - F4 loopback-only bind is enforced at lifespan.py (port config), not here
  - F2 runtime Falkor ping (in-process: ``ctx.supervisor.probe()`` pre-call)

What this handler does NOT do:

  - NOT do client-side diacritic dual-query. Engine has the canonical
    contract: ``cfg.search.diacritic_dual_query=true`` opts in; otherwise
    the hook is the canonical source and emits dual-string. PicoClaw
    hook owns diacritic policy; engine validates fail-loud.

Cancellation (F8): aiohttp auto-cancels the handler task on client
disconnect. We do NOT add speculative cancellation code (verified by
``test_disconnect_cancels_handler``).

Error envelope:
  200 + {success, group_id, query, count, results, elapsed_ms, degraded}
  400 INVALID_QUERY / INVALID_LIMIT
  422 DUAL_QUERY_REQUIRED (only when cfg.search.diacritic_dual_query=true)
  429 RATE_LIMITED
  503 SEARCH_BACKEND_DOWN / FALKOR_DISCONNECTED / NOT_READY
  504 SEARCH_TIMEOUT (handler-side or engine-reported BACKEND_TIMEOUT)
  500 INTERNAL_ERROR
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Deque, Dict

from aiohttp import web

from ..engine.search_memory import search_memory as engine_search_memory

if TYPE_CHECKING:
    from .lifespan import DaemonContext

logger = logging.getLogger("janus_graph.daemon.search_memory_handler")

# PR 2 / v0.5.0 F5: rate limit threshold. 60 req/min/IP matches PicoClaw
# hook 3-path budget (HTTP_BUDGET_MS=3000 + MCP_BUDGET_MS=5500 = 8.5s per
# recall × ~7 parallel recalls = ~60 RPM theoretical upper bound for
# bursty clients). Set here as a module constant; if we need per-IP
# dynamic limits, promote to config later.
_RATE_LIMIT_REQUESTS = 60
_RATE_LIMIT_WINDOW_SEC = 60.0

# PR 2 / v0.5.0: input validation thresholds. Mirror hook-side guards
# (``mcp_recall.js:query``) so HTTP callers can't bypass the policy.
_MIN_QUERY_LEN = 4   # hook enforces >=4 to filter out noise like "hi"
_MAX_QUERY_LEN = 512  # cap before graphiti embed (avoid OOM on text blob)
_DEFAULT_LIMIT = 5
_MIN_LIMIT = 1
_MAX_LIMIT = 50

# v0.5.0.1 fix #1: bounded timeout around the engine call. Mirrors F6
# boot-time pattern: same rationale (Falkor disconnect mid-call would
# otherwise hang the handler forever). 8s cap > measure baseline (warm
# graphiti_search ~2-4s; hook HTTP_BUDGET_MS=3000); < ratio that would
# let a single slow recall exhaust aiohttp handler concurrency. TimeoutError
# maps to 504 SEARCH_TIMEOUT (symmetric with /search/graph endpoint).
_ENGINE_CALL_TIMEOUT_SEC = 8.0


# ─── Rate limit state (F5) ──────────────────────────────────────────────


class _TokenBucket:
    """In-process sliding-window rate limiter, per remote IP.

    F5: hand-rolled dict+deque (no external dep). Trades memory for
    simplicity — at 60 RPM/IP and a small PicoClaw client surface, the
    state is bounded to a few dozen IPs.

    Thread-safety: aiohttp runs handlers on a single event loop; we
    never await inside ``consume()`` so no lock is needed.

    v0.5.0.1 fix #6: bounded memory for adversarial IPs. Without a cap,
    a single attacker hitting the endpoint from a botnet could grow
    ``_hits`` unbounded. Two guards:
      1. ``_MAX_KEYS`` (default 10_000) — if a NEW IP arrives when the
         dict already has this many keys, evict the OLDEST entry (LRU
         proxy via insertion order). PicoClaw's known client surface
         is ~10s, so 10K is a 1000x over-budget safety.
      2. ``_maxlen`` on each deque is set to ``max_requests`` so a
         bucket that hits cap can't grow further; the eager-evict
         loop in ``consume()`` keeps it at exactly max_requests
         during steady-state.
    """

    # v0.5.0.1 fix #6: max distinct IPs we'll track before LRU-eviction.
    # Conservative — actual PicoClaw hook surface is <20 IPs.
    _MAX_KEYS = 10_000

    def __init__(self, max_requests: int, window_sec: float) -> None:
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: "Dict[str, Deque[float]]" = defaultdict(
            lambda: deque(maxlen=max_requests)
        )
        # v0.5.0.1 fix #6: insertion order proxy (used by LRU eviction).
        # We can't rely on ``_hits`` ordering because defaultdict
        # doesn't move-to-end on access. Use a parallel ordered set.
        self._keys_in_order: "Dict[str, None]" = {}

    def _evict_oldest(self) -> None:
        """LRU evict the oldest-inserted key. O(1) amortized.

        Removes the key from BOTH the tracking dict (``_keys_in_order``)
        AND the hits dict (``_hits``). We unconditionally drop the hits
        deque even if it has active rate-limit state — the alternative
        would be to leak entries (defeating the cap) or scan every key
        for emptiness (defeating O(1)). Under adversarial load, LRU
        eviction of an active IP is preferable to unbounded memory.
        """
        if not self._keys_in_order:
            return
        oldest_key = next(iter(self._keys_in_order))
        del self._keys_in_order[oldest_key]
        # Always drop from _hits — see docstring rationale. ``pop`` with
        # default to avoid KeyError if the entry was already removed
        # by some other path (defensive; shouldn't happen in practice).
        self._hits.pop(oldest_key, None)

    def consume(self, key: str, now: float) -> bool:
        """Record a hit for ``key`` if within budget.

        Returns True if request is allowed, False if rate-limited.
        Side effect: evicts hits older than ``window_sec`` from the
        deque. Backward-compat: this method is sync because the
        ``now`` is passed in (caller controls clock for tests).
        """
        # v0.5.0.1 fix #6: bound the dict size. If we're at cap and
        # this is a NEW key (not just a refresh), evict the oldest.
        if key not in self._hits and len(self._hits) >= self._MAX_KEYS:
            self._evict_oldest()
        self._keys_in_order[key] = None

        bucket = self._hits[key]
        cutoff = now - self.window_sec
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def reset(self) -> None:
        """Test hook: clear all per-IP state."""
        self._hits.clear()
        self._keys_in_order.clear()


# Module-level singleton; lifespan.py does NOT manage this (intentionally
# — process restart = clean state, no need to plumb through DaemonContext).
_RATE_LIMITER = _TokenBucket(
    max_requests=_RATE_LIMIT_REQUESTS,
    window_sec=_RATE_LIMIT_WINDOW_SEC,
)


# ─── Handler ────────────────────────────────────────────────────────────


async def search_memory_handler(request: web.Request) -> web.Response:
    """GET /search/memory — knowledge-graph free-text recall.

    Query params:
      query  (str, required, len 4..512)
      limit  (int, optional, default 5, range [1, 50])

    Responses:
      200 + {success, group_id, query, count, results, elapsed_ms, degraded}
      400 INVALID_QUERY / INVALID_LIMIT
      422 DUAL_QUERY_REQUIRED (when cfg.search.diacritic_dual_query=true
          and engine detects a single-string query with diacritics)
      429 RATE_LIMITED + Retry-After header
      503 SEARCH_BACKEND_DOWN / FALKOR_DISCONNECTED / NOT_READY
      500 INTERNAL_ERROR
    """
    ctx: "DaemonContext" = request.app["ctx"]

    # ── F5: rate limit per remote IP ──────────────────────────────────
    remote = request.remote or "unknown"
    if not _RATE_LIMITER.consume(remote, time.monotonic()):
        logger.info("/search/memory rate-limited ip=%s", remote)
        return web.json_response(
            {
                "code": "RATE_LIMITED",
                "message": f"max {_RATE_LIMIT_REQUESTS} req/"
                           f"{int(_RATE_LIMIT_WINDOW_SEC)}s per IP",
            },
            status=429,
            headers={"Retry-After": str(int(_RATE_LIMIT_WINDOW_SEC))},
        )

    # ── Query param parsing ───────────────────────────────────────────
    raw_query = request.query.get("query")
    if not raw_query or not isinstance(raw_query, str):
        return web.json_response(
            {"code": "INVALID_QUERY", "message": "query param required"},
            status=400,
        )
    query = raw_query.strip()
    if len(query) < _MIN_QUERY_LEN or len(query) > _MAX_QUERY_LEN:
        return web.json_response(
            {
                "code": "INVALID_QUERY",
                "message": f"query length must be in "
                           f"[{_MIN_QUERY_LEN}, {_MAX_QUERY_LEN}]",
            },
            status=400,
        )

    raw_limit = request.query.get("limit", str(_DEFAULT_LIMIT))
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return web.json_response(
            {"code": "INVALID_LIMIT", "message": "limit must be an integer"},
            status=400,
        )
    if limit < _MIN_LIMIT or limit > _MAX_LIMIT:
        return web.json_response(
            {
                "code": "INVALID_LIMIT",
                "message": f"limit must be in [{_MIN_LIMIT}, {_MAX_LIMIT}]",
            },
            status=400,
        )

    # ── Backend readiness (mirrors /search/graph contract) ────────────
    graphiti = getattr(ctx, "graphiti_instance", None)
    if graphiti is None:
        return web.json_response(
            {
                "code": "SEARCH_BACKEND_DOWN",
                "message": "graphiti singleton not initialized "
                           "(see daemon boot logs)",
            },
            status=503,
        )

    # ── F2 runtime Falkor disconnect detection ────────────────────────
    # Cheap pre-flight: ping Falkor with bounded timeout. If the
    # supervisor's breaker is OPEN, ``probe()`` returns False quickly.
    try:
        falkor_ok = await asyncio.wait_for(
            ctx.supervisor.probe(), timeout=1.0
        )
    except asyncio.TimeoutError:
        falkor_ok = False
    if not falkor_ok:
        return web.json_response(
            {
                "code": "FALKOR_DISCONNECTED",
                "message": "Falkor unreachable; retry after circuit resets",
            },
            status=503,
        )

    # ── Delegate to engine facade (single source of truth) ────────────
    started = time.monotonic()
    try:
        # v0.5.0.1 fix #1: bounded engine call. Falkor disconnect after
        # probe OK but before graphiti_search returns → handler would
        # hang otherwise. 8s cap mirrors F6 boot pattern (timeout=10s).
        # CancelledError: aiohttp auto-cancels on client disconnect; we
        # let it propagate per F8 (test_no_speculative_cancellation_code).
        result = await asyncio.wait_for(
            engine_search_memory(
                ctx.settings, query, limit, graphiti=graphiti,
            ),
            timeout=_ENGINE_CALL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        logger.warning(
            "/search/memory timed out after %.1fs query=%r limit=%d",
            _ENGINE_CALL_TIMEOUT_SEC, query, limit,
        )
        return web.json_response(
            {
                "code": "SEARCH_TIMEOUT",
                "message": f"search exceeded {int(_ENGINE_CALL_TIMEOUT_SEC)}s budget",
                "elapsed_ms": elapsed_ms,
            },
            status=504,
            headers={"Retry-After": "5"},
        )
    except Exception as e:  # noqa: BLE001 — defensive envelope
        # Engine already catches its own errors and returns
        # {"success": False, ...}. This ``except`` only fires for
        # bugs in the handler envelope itself; preserve the contract.
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        logger.exception(
            "/search/memory envelope failure query=%r limit=%d",
            query, limit,
        )
        return web.json_response(
            {
                "code": "INTERNAL_ERROR",
                "message": str(e),
                "elapsed_ms": elapsed_ms,
            },
            status=500,
        )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)

    # ── Map engine result → HTTP response ─────────────────────────────
    if not result.get("success"):
        # Engine returned error envelope (e.g., graphiti.search raised).
        # v0.5.0.1 fix #4: prefer the typed ``code`` field over fragile
        # substring matching on the human-readable ``error`` string.
        # Engine contract: emit ``code`` ∈ {"FALKOR_DISCONNECTED",
        # "BACKEND_TIMEOUT", "INTERNAL_ERROR"} — handler maps by code,
        # not by string fragment. Old substring match kept as fallback
        # for engine versions < v0.5.0.1 (MCP path may emit plain str).
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        engine_code = result.get("code")
        err = result.get("error", "unknown")
        if engine_code == "FALKOR_DISCONNECTED":
            return web.json_response(
                {
                    "code": "FALKOR_DISCONNECTED",
                    "message": err,
                    "elapsed_ms": elapsed_ms,
                },
                status=503,
            )
        if engine_code == "BACKEND_TIMEOUT":
            # Engine-level timeout (e.g., graphiti_search internal cap)
            # → surface as 504, distinct from handler-side 504 above.
            return web.json_response(
                {
                    "code": "SEARCH_TIMEOUT",
                    "message": err,
                    "elapsed_ms": elapsed_ms,
                },
                status=504,
                headers={"Retry-After": "5"},
            )
        # Fallback for engine returns without ``code`` (pre-v0.5.0.1):
        # F2 detection via string fragment. Fragile by design; will be
        # removed once all engine callers emit the typed code.
        if any(token in err.lower() for token in ("falkor", "redis", "disconnect", "connection")):
            return web.json_response(
                {
                    "code": "FALKOR_DISCONNECTED",
                    "message": err,
                    "elapsed_ms": elapsed_ms,
                },
                status=503,
            )
        return web.json_response(
            {
                "code": "INTERNAL_ERROR",
                "message": err,
                "elapsed_ms": elapsed_ms,
            },
            status=500,
        )

    payload = {
        "success": True,
        "group_id": result.get("group_id"),
        "query": result.get("query", query),
        "count": result.get("count", 0),
        "results": result.get("results", []),
        "elapsed_ms": elapsed_ms,
        # v0.5.0.1 fix #5: degraded is always False in v0.5.0. Falkor
        # disconnect is REJECTED at the handler with 503 before this
        # payload is built — there's no "partial result" path here, so
        # the field mirrors /search/graph semantics (also always False).
        # Future work: if we ever expose a partial-result path (e.g.,
        # cache fallback when graphiti is alive but slow), wire this
        # from SearchEngine; until then treat as a constant.
        "degraded": False,
    }
    return web.json_response(payload, status=200)
