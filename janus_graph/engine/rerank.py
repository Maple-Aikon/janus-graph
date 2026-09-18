"""Engine rerank module — optional BGE-reranker post-MMR reordering.

Contract:
    Called by janus_graph.engine.search_memory AFTER the MMR-ranked facts
    list is materialised (F10) and BEFORE the return dict. The rerank is
    best-effort: any failure (timeout, circuit OPEN, malformed payload,
    transport error) degrades to the original MMR ordering. Never raises.

    PicoClaw hook forward-compat note (F23):
        When the PicoClaw daemon hook flips its HTTP recall path
        (``hooks/lib/mcp_recall_http.js``) from stdio MCP to the daemon
        HTTP ``/search/memory`` endpoint, this same rerank pass will be
        applied there too — the engine facade is the single SoT.

Circuit-breaker state machine (CLOSED / OPEN / HALF_OPEN) follows the
pattern used by FalkorCircuitSettings — see ``janus_graph.daemon.circuit_breaker``.

F22 (deferred to v2.1): expose ``state`` on the daemon ``/health`` endpoint
as ``rerank_circuit_state``. See plan
``wire-bge-reranker-into-search-memory-v3-...-20260918.md`` §F22 — no
change to the daemon HTTP surface in this PR. TODO: add a small helper
``def get_circuit_state() -> str`` and wire into the handler.

Lock discipline (F11/F24/F26):
    * ``_circuit_init_lock`` protects the LAZY INIT of the singleton
      ``_circuit`` (F26). Caller HOLDS it during the entire
      allow_request → HTTP → record_* block. ``asyncio.Lock`` is NOT
      re-entrant, so internal helpers MUST NOT re-enter it.
    * The HTTP call itself runs INSIDE the lock, which serialises
      concurrent rerank requests (acceptable: rerank is a single
      /v1/rerank call per search, latency ~50-200ms).
    * ``record_success`` / ``record_failure`` are called WITH the lock
      held by the caller — they DO NOT take the lock themselves.

Validation order (F27):
    ``_validate_and_sort`` builds ``valid_results: list[tuple[int, float]]``
    FIRST by iterating the response, skipping malformed entries (F15), then
    sorts the validated list. Sorting FIRST would KeyError/TypeError on a
    single malformed relevance_score and abort the whole rerank.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import SecretStr

logger = logging.getLogger("janus_graph.engine.rerank")

# Calibration: ~60ms per document on q8_0 BGE-reranker-v2-m3 at llama-server
# :8083 (F25). Conservative upper-bound; bump if you measure otherwise.
_PER_DOC_SEC: float = 0.06

# F25: timeout minimum floor. Even single-doc rerank should not have
# sub-second timeout (avoids spurious timeout on warm-startup cold cache).
_TIMEOUT_MIN_SEC: float = 1.0


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class _RerankCircuit:
    """Three-state circuit breaker for the rerank HTTP backend.

    Mirrors the FalkorCircuitSettings semantics (CLOSED → OPEN at
    ``failure_threshold`` consecutive failures, OPEN → HALF_OPEN after
    ``reset_timeout_sec``).
    """

    failure_threshold: int = 3
    reset_timeout_sec: float = 30.0
    half_open_max_probes: int = 1

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    opened_at: Optional[float] = None
    half_open_probes: int = 0

    def allow_request(self) -> bool:
        """Decide whether the next request should be allowed.

        Caller MUST hold ``_circuit_init_lock`` when calling this.
        """
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            # F20: defensive opened_at is None — treat as fresh trip
            if self.opened_at is None:
                self.state = CircuitState.HALF_OPEN
                self.half_open_probes = 0
                return True
            if (time.monotonic() - self.opened_at) >= self.reset_timeout_sec:
                # Time-based reset → HALF_OPEN with fresh probe budget
                self.state = CircuitState.HALF_OPEN
                self.half_open_probes = 0
                return True
            return False
        # HALF_OPEN
        if self.half_open_probes < self.half_open_max_probes:
            self.half_open_probes += 1
            return True
        return False

    def record_success(self) -> None:
        """Reset to CLOSED on any success.

        Caller MUST hold ``_circuit_init_lock`` when calling this.
        """
        if self.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info("rerank circuit closed after success")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = None
        self.half_open_probes = 0

    def record_failure(self) -> None:
        """Increment failure count, trip OPEN at threshold.

        Caller MUST hold ``_circuit_init_lock`` when calling this.
        """
        self.failure_count += 1
        if self.state is CircuitState.HALF_OPEN:
            # Probe failed — straight back to OPEN with fresh window
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning("rerank circuit re-opened after HALF_OPEN probe failure")
            return
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(
                "rerank circuit opened after %d consecutive failures",
                self.failure_count,
            )


# Module-level singletons (lazy-initialised; protected by init lock)
_circuit_init_lock: asyncio.Lock = asyncio.Lock()
_circuit: Optional[_RerankCircuit] = None

_session: Optional[aiohttp.ClientSession] = None  # F13

# F28: module-level flag caches the unknown-env warning so we only log once
# per process lifetime (not per RerankConfig construction).
_WARNED_UNKNOWN_ENV: bool = False


def _scan_unknown_env() -> List[str]:
    """Return list of JANUS_SEARCH__RERANK__* env vars that don't map to a known field.

    F14: pydantic ``extra=\"ignore\"`` silently DROPS unknown env keys, which is
    the worst possible failure mode for ops typos (e.g. JANUS_SEARCH__RERANK__
    vs JANUS_SEARCH__RERANK__). We scan os.environ at first RerankConfig
    construction and emit a UserWarning (visible in pytest -W default + CI).
    """
    known = {
        "JANUS_SEARCH__RERANK__ENABLED",
        "JANUS_SEARCH__RERANK__BASE_URL",
        "JANUS_SEARCH__RERANK__MODEL_NAME",
        "JANUS_SEARCH__RERANK__API_KEY",
        "JANUS_SEARCH__RERANK__TIMEOUT_SEC",
        "JANUS_SEARCH__RERANK__MAX_CANDIDATES",
        "JANUS_SEARCH__RERANK__FAILURE_THRESHOLD",
        "JANUS_SEARCH__RERANK__RESET_TIMEOUT_SEC",
    }
    return [
        key for key in os.environ
        if key.startswith("JANUS_SEARCH__RERANK__") and key not in known
    ]


def _derive_timeout(rerank_cfg: Any) -> float:
    """F25: scale timeout by max_candidates so broad queries don't time out.

    Formula: ``max(timeout_sec, max_candidates * _PER_DOC_SEC)`` floored at
    ``_TIMEOUT_MIN_SEC``. Single-doc rerank keeps the configured
    ``timeout_sec``; 20-doc rerank gets max(2.0, 1.2) = 2.0s; 50-doc
    gets max(2.0, 3.0) = 3.0s.

    Calibration note: _PER_DOC_SEC is an estimate (60ms/doc on q8_0 BGE
    at :8083). Probe and tune if you observe systematic over/under-timeout.
    """
    return max(
        _TIMEOUT_MIN_SEC,
        float(rerank_cfg.timeout_sec),
        int(rerank_cfg.max_candidates) * _PER_DOC_SEC,
    )


def _validate_and_sort(
    raw_results: List[Any],
    top_n: int,
) -> List[Dict[str, Any]]:
    """F27: validate FIRST, sort SECOND.

    Returns the top-N results sorted by ``relevance_score`` descending.
    Malformed entries (missing ``index`` / ``relevance_score``) are
    SKIPPED with a warning (F15). Sorting before validation would
    KeyError/TypeError on a single bad score and abort the entire
    rerank — wrong semantics for a best-effort post-MMR pass.
    """
    valid: List[Tuple[int, float]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            logger.warning("rerank: skipping non-dict entry %r", entry)
            continue
        idx = entry.get("index")
        score = entry.get("relevance_score")
        if not isinstance(idx, int) or not isinstance(score, (int, float)):
            logger.warning("rerank: skipping malformed entry %r", entry)
            continue
        valid.append((idx, float(score)))

    # Sort AFTER validation — see F27
    valid.sort(key=lambda x: -x[1])
    return [{"index": i, "relevance_score": s} for i, s in valid[:top_n]]


async def _get_circuit(cfg: Any) -> _RerankCircuit:
    """F26: lazy-initialise the singleton ``_circuit`` from cfg.

    Caller MUST hold ``_circuit_init_lock``. Re-checks ``_circuit is None``
    after acquiring to handle the concurrent-init race (two coroutines
    enter the awaitable before either finishes init).
    """
    global _circuit
    if _circuit is not None:
        return _circuit
    rerank_cfg = cfg.search.rerank
    _circuit = _RerankCircuit(
        failure_threshold=int(rerank_cfg.failure_threshold),
        reset_timeout_sec=float(rerank_cfg.reset_timeout_sec),
        half_open_max_probes=1,
    )
    return _circuit


async def _get_session() -> aiohttp.ClientSession:
    """F13: module-level session, created lazily, reused across calls.

    Per-call session creation costs ~50-100ms TCP/TLS handshake —
    unacceptable for a per-search pass. We use a single session for the
    process lifetime; cleanup is the caller's job (daemon lifespan).
    """
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0),
            connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300),
        )
    return _session


def warn_unknown_env_once() -> List[str]:
    """F14 + F28: scan os.environ, warn once per process.

    Returns the unknown-key list so tests can assert on it. Idempotent
    via the ``_WARNED_UNKNOWN_ENV`` module-level flag.
    """
    global _WARNED_UNKNOWN_ENV
    unknown = _scan_unknown_env()
    if unknown and not _WARNED_UNKNOWN_ENV:
        warnings.warn(
            f"Unknown JANUS_SEARCH__RERANK__* env vars (ignored): {unknown}",
            UserWarning,
            stacklevel=2,
        )
        _WARNED_UNKNOWN_ENV = True
    return unknown


async def _maybe_rerank(
    cfg: Any,
    query: str,
    facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """F10 + F11 + F24: optional BGE-reranker post-MMR reordering.

    Lock discipline:
        ``async with _circuit_init_lock`` wraps the ENTIRE
        allow_request → HTTP → record_* sequence. ``asyncio.Lock`` is
        NOT re-entrant — the helper record_success / record_failure
        methods MUST NOT take this lock themselves. This serialises
        concurrent rerank requests (acceptable trade-off: rerank is
        single-call per search, ~50-200ms latency).

    Failure modes (all degrade to original MMR ordering):
        * cfg.search.rerank.enabled is False → return facts unchanged
        * circuit OPEN → return facts unchanged
        * HTTP timeout / connection error → record_failure, return facts
        * malformed response → log+skip, return facts unchanged
        * fewer valid results than top_n → return what we have
    """
    rerank_cfg = cfg.search.rerank
    warn_unknown_env_once()  # F14 + F28

    if not rerank_cfg.enabled:
        return facts
    if not facts:
        return facts

    # Cap input at max_candidates — broader input would exceed timeout
    candidates = facts[: int(rerank_cfg.max_candidates)]
    top_n = min(len(facts), len(candidates))
    timeout_sec = _derive_timeout(rerank_cfg)

    payload = {
        "model": rerank_cfg.model_name or "",  # F21: empty allowed
        "query": query,
        "documents": [c.get("fact", "") for c in candidates],
    }
    headers: Dict[str, str] = {}
    if rerank_cfg.api_key is not None:
        # F12: SecretStr → reveal just-in-time, never log
        headers["Authorization"] = f"Bearer {rerank_cfg.api_key.get_secret_value()}"

    url = rerank_cfg.base_url.rstrip("/") + "/v1/rerank"

    async with _circuit_init_lock:  # F11 + F24 + F26
        circuit = await _get_circuit(cfg)
        if not circuit.allow_request():
            # Circuit OPEN — degrade silently
            return facts

        # HTTP call INSIDE lock (serialised). Acceptable: one /v1/rerank
        # per search, ~50-200ms latency.
        session = await _get_session()
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as resp:
                if resp.status != 200:
                    logger.warning("rerank: HTTP %d, falling back", resp.status)
                    circuit.record_failure()
                    return facts
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("rerank: transport error %r, falling back", e)
            circuit.record_failure()
            return facts

        # Validate response shape
        raw_results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw_results, list):
            logger.warning("rerank: missing/invalid 'results' field")
            circuit.record_failure()
            return facts

        try:
            ranked = _validate_and_sort(raw_results, top_n)  # F27
        except Exception as e:  # noqa: BLE001 — last-line guard
            logger.warning("rerank: validate_and_sort raised %r", e)
            circuit.record_failure()
            return facts

        if not ranked:
            # Empty valid_results — treat as no-op (don't open the circuit)
            return facts

        circuit.record_success()

    # Reorder facts per ranked indices (OUTSIDE the lock)
    index_to_fact: Dict[int, Dict[str, Any]] = {
        idx: candidates[idx] for idx in range(len(candidates))
    }
    reordered: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for entry in ranked:
        idx = entry["index"]
        if idx in index_to_fact and idx not in seen:
            reordered.append(index_to_fact[idx])
            seen.add(idx)
    # Append any facts beyond max_candidates (not reranked) at the tail
    for i, f in enumerate(facts):
        if i not in seen:
            reordered.append(f)
            seen.add(i)
    return reordered


def reset_for_test() -> None:
    """Test helper: reset all module singletons + warn flag.

    Tests should call this in a fixture setup to ensure isolation
    across cases (especially F26 concurrent-init and F28 warn-once).
    """
    global _circuit, _session, _WARNED_UNKNOWN_ENV
    _circuit = None
    _session = None
    _WARNED_UNKNOWN_ENV = False
