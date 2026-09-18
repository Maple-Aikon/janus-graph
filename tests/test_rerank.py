"""Tests for ``janus_graph.engine.rerank`` module.

Plan: ``wire-bge-reranker-into-search-memory-v3-...-20260918.md``
Covers F10-F28 folded in v3.1:

* F10 — integration site patch (call site exists, returns sorted)
* F11 — narrow lock scope (no internal lock in record_*; caller holds)
* F12 — SecretStr api_key redacted in str/repr
* F13 — module-level aiohttp.ClientSession reused (not per-call)
* F14 — model_validator warns unknown JANUS_SEARCH__RERANK__* env
* F15 — skip malformed results[index] entries
* F19 — Risks cite re-stated to actual line ranges
* F20 — HALF_OPEN defensive opened_at is None check
* F21 — model_name="" handled (llama-server ignores)
* F22 — TODO marker present (deferred to v2.1)
* F23 — PicoClaw hook forward-compat 1-liner present
* F24 — concurrent parallel: 5 tasks < 10s cap
* F25 — timeout scales with max_candidates
* F26 — env failure_threshold + concurrent _get_circuit init
* F27 — validate-first then sort
* F28 — warn-once via _WARNED_UNKNOWN_ENV

Test counts: 22 unit + 4 integration = 26 tests.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from janus_graph.config import JanusSettings, RerankConfig
from janus_graph.engine import rerank as rerank_mod
from janus_graph.engine.rerank import (
    CircuitState,
    _circuit_init_lock,
    _derive_timeout,
    _get_circuit,
    _get_session,
    _maybe_rerank,
    _PER_DOC_SEC,
    _RerankCircuit,
    _scan_unknown_env,
    _TIMEOUT_MIN_SEC,
    _validate_and_sort,
    _WARNED_UNKNOWN_ENV,
    reset_for_test,
    warn_unknown_env_once,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """F26 + F28: reset module singletons between tests for isolation."""
    reset_for_test()
    yield
    reset_for_test()


def _make_fake_session(*, status: int = 200, json_payload=None, delay: float = 0.0):
    """Build a fake aiohttp-style session for mocking _get_session.

    aiohttp's ``session.post(url, ...)`` returns a ``_RequestContextManager``
    (an awaitable async-CM) DIRECTLY — NOT a coroutine. The caller then
    does ``async with session.post(...) as resp:`` which awaits the CM and
    enters its ``__aenter__``. Our mock must mirror that shape.
    """
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_payload or {})

    session = MagicMock()
    session.closed = False

    def fake_post(*args, **kwargs):
        # _RequestContextManager is awaitable + supports async with.
        # MagicMock by default supports __await__ + __aenter__/__aexit__.
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        if delay:
            # Insert a sleep coroutine via __await__
            async def _await_with_delay():
                await asyncio.sleep(delay)
                return cm
            cm.__await__ = lambda: _await_with_delay().__await__()
        return cm

    session.post = fake_post
    return session


# ─── F25: timeout derivation ──────────────────────────────────────────────


class TestF25TimeoutDerivation:
    """F25: timeout must scale with max_candidates so broad queries don't
    systematically time out on the queries rerank was meant for.
    """

    def test_f25_single_doc_returns_floor(self):
        cfg = RerankConfig(max_candidates=1, timeout_sec=2.0)
        # max(1.0, 2.0, 0.06) = 2.0
        assert abs(_derive_timeout(cfg) - 2.0) < 1e-6

    def test_f25_twenty_docs_returns_configured(self):
        cfg = RerankConfig(max_candidates=20, timeout_sec=2.0)
        # max(1.0, 2.0, 1.2) = 2.0
        assert abs(_derive_timeout(cfg) - 2.0) < 1e-6

    def test_f25_fifty_docs_scales_up(self):
        cfg = RerankConfig(max_candidates=50, timeout_sec=2.0)
        # max(1.0, 2.0, 3.0) = 3.0
        assert abs(_derive_timeout(cfg) - 3.0) < 1e-6

    def test_f25_floor_applies_when_both_below(self):
        cfg = RerankConfig(max_candidates=1, timeout_sec=0.5)
        # max(1.0, 0.5, 0.06) = 1.0 (floor wins)
        assert abs(_derive_timeout(cfg) - 1.0) < 1e-6

    def test_f25_calibration_constant_documented(self):
        # If someone tunes _PER_DOC_SEC, the test reminds them to update
        # the docstring + plan §F25 calibration note.
        assert _PER_DOC_SEC == 0.06
        assert _TIMEOUT_MIN_SEC == 1.0


# ─── F27: validate-first then sort ────────────────────────────────────────


class TestF27ValidateFirstSort:
    """F27: build valid_results FIRST, sort SECOND.

    Sorting first would KeyError/TypeError on a single malformed
    relevance_score and abort the entire rerank — wrong semantics for
    a best-effort post-MMR pass.
    """

    def test_f27_basic_sort_descending(self):
        results = [
            {"index": 0, "relevance_score": 1.0},
            {"index": 1, "relevance_score": 5.0},
            {"index": 2, "relevance_score": 3.0},
        ]
        sorted_r = _validate_and_sort(results, top_n=3)
        assert [r["index"] for r in sorted_r] == [1, 2, 0]
        assert [r["relevance_score"] for r in sorted_r] == [5.0, 3.0, 1.0]

    def test_f27_skip_malformed_missing_index(self):
        # F15: skip entries with missing/invalid fields
        results = [
            {"index": 0, "relevance_score": 5.0},
            {"relevance_score": 9.0},  # missing index
            {"index": 1, "relevance_score": 3.0},
        ]
        sorted_r = _validate_and_sort(results, top_n=3)
        assert len(sorted_r) == 2
        assert [r["index"] for r in sorted_r] == [0, 1]

    def test_f27_skip_non_dict(self):
        results = [
            {"index": 0, "relevance_score": 5.0},
            "bad-entry",
            None,
            {"index": 1, "relevance_score": 3.0},
        ]
        sorted_r = _validate_and_sort(results, top_n=3)
        assert len(sorted_r) == 2

    def test_f27_skip_wrong_types(self):
        results = [
            {"index": "zero", "relevance_score": 5.0},  # idx not int
            {"index": 0, "relevance_score": "high"},     # score not numeric
            {"index": 1, "relevance_score": 3.0},
        ]
        sorted_r = _validate_and_sort(results, top_n=3)
        assert len(sorted_r) == 1
        assert sorted_r[0]["index"] == 1

    def test_f27_top_n_limits_output(self):
        results = [
            {"index": i, "relevance_score": float(i)}
            for i in range(10)
        ]
        sorted_r = _validate_and_sort(results, top_n=3)
        assert len(sorted_r) == 3
        assert [r["index"] for r in sorted_r] == [9, 8, 7]


# ─── F11: circuit state machine + lock scope ─────────────────────────────


class TestF11CircuitAndLock:
    """F11: ``record_success`` / ``record_failure`` MUST NOT take
    ``_circuit_init_lock`` themselves — asyncio.Lock is not re-entrant.
    Caller HOLDS the lock for the entire allow_request → HTTP → record_*
    block. F20: defensive opened_at is None check on OPEN state.
    """

    def test_f11_record_methods_take_no_lock(self):
        """F11: source-level guarantee — no `async with _circuit_init_lock`
        inside record_success / record_failure. We use AST inspection (not
        substring) because the docstring mentions ``_circuit_init_lock``."""
        import ast
        import textwrap

        for method_name in ("record_success", "record_failure", "allow_request"):
            method = getattr(_RerankCircuit, method_name)
            src = textwrap.dedent(inspect.getsource(method))
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncWith):
                    for item in node.items:
                        ctx_id = getattr(item.context_expr, "id", None)
                        if ctx_id == "_circuit_init_lock":
                            raise AssertionError(
                                f"F11: {method_name} must not take _circuit_init_lock"
                            )

    def test_f11_allow_request_takes_no_lock(self):
        """F11: allow_request must NOT take the lock — caller holds it."""
        import ast
        import textwrap

        src = textwrap.dedent(inspect.getsource(_RerankCircuit.allow_request))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncWith):
                for item in node.items:
                    if getattr(item.context_expr, "id", None) == "_circuit_init_lock":
                        raise AssertionError(
                            "F11: allow_request must not take _circuit_init_lock"
                        )

    def test_f11_closed_state_allows_traffic(self):
        c = _RerankCircuit(failure_threshold=3)
        assert c.allow_request() is True
        assert c.state is CircuitState.CLOSED

    def test_f11_threshold_trips_open(self):
        c = _RerankCircuit(failure_threshold=2)
        c.record_failure()
        assert c.state is CircuitState.CLOSED
        c.record_failure()
        assert c.state is CircuitState.OPEN
        assert c.allow_request() is False

    def test_f11_success_resets_to_closed(self):
        c = _RerankCircuit(failure_threshold=1)
        c.record_failure()
        assert c.state is CircuitState.OPEN
        # Mock the time check so we don't actually have to wait
        c.opened_at = time.monotonic() - 100
        assert c.allow_request() is True
        c.record_success()
        assert c.state is CircuitState.CLOSED
        assert c.failure_count == 0


class TestF20HalfOpenDefensive:
    """F20: if opened_at is None (corrupted state), treat as fresh trip."""

    def test_f20_opened_at_none_resets_to_half_open(self):
        c = _RerankCircuit(failure_threshold=1)
        c.state = CircuitState.OPEN
        c.opened_at = None  # corrupt
        assert c.allow_request() is True
        assert c.state is CircuitState.HALF_OPEN

    def test_f20_half_open_probe_failure_reopens(self):
        c = _RerankCircuit(failure_threshold=1, half_open_max_probes=1)
        c.state = CircuitState.OPEN
        c.opened_at = None
        # Probe allowed
        assert c.allow_request() is True
        assert c.state is CircuitState.HALF_OPEN
        # Probe fails → straight back to OPEN
        c.record_failure()
        assert c.state is CircuitState.OPEN
        assert c.opened_at is not None


# ─── F12: SecretStr redaction ─────────────────────────────────────────────


class TestF12SecretRedaction:
    """F12: SecretStr api_key must NEVER appear in repr/str/log.
    Use SecretStr so pydantic's default repr also redacts (defense in depth).
    """

    def test_f12_repr_redacts_api_key(self):
        cfg = RerankConfig(api_key="supersecret-token-12345")
        r = repr(cfg)
        assert "supersecret-token" not in r
        assert "***" in r

    def test_f12_repr_shows_none_when_unset(self):
        cfg = RerankConfig()
        r = repr(cfg)
        assert "api_key=None" in r

    def test_f12_secret_value_revealed_via_method_only(self):
        from pydantic import SecretStr
        # Use pydantic SecretStr directly to confirm reveal path
        s = SecretStr("my-token")
        assert s.get_secret_value() == "my-token"
        assert "my-token" not in str(s)
        assert "my-token" not in repr(s)


# ─── F14: warn unknown env ────────────────────────────────────────────────


class TestF14UnknownEnvWarning:
    """F14: pydantic ``extra=\"ignore\"`` silently DROPS unknown env keys.
    We scan os.environ at first RerankConfig construction and warn."""

    def test_f14_scan_finds_unknown_keys(self, monkeypatch):
        monkeypatch.setenv("JANUS_SEARCH__RERANK__ENABLED", "true")
        monkeypatch.setenv("JANUS_SEARCH__RERANK__TYPO_KEY", "x")
        unknown = _scan_unknown_env()
        assert "JANUS_SEARCH__RERANK__TYPO_KEY" in unknown
        assert "JANUS_SEARCH__RERANK__ENABLED" not in unknown

    def test_f14_scan_empty_when_only_known(self, monkeypatch):
        # Clear all JANUS_SEARCH__RERANK__* then set only known
        for k in [k for k in os.environ if k.startswith("JANUS_SEARCH__RERANK__")]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("JANUS_SEARCH__RERANK__ENABLED", "true")
        assert _scan_unknown_env() == []


# ─── F28: warn-once ───────────────────────────────────────────────────────


class TestF28WarnOnce:
    """F28: warn exactly once per process — module flag caches."""

    def test_f28_warns_first_time(self, monkeypatch):
        monkeypatch.setenv("JANUS_SEARCH__RERANK__TYPO_KEY", "x")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unknown = warn_unknown_env_once()
            assert "JANUS_SEARCH__RERANK__TYPO_KEY" in unknown
            assert any("Unknown JANUS_SEARCH__RERANK__*" in str(w.message) for w in caught)

    def test_f28_does_not_warn_second_time(self, monkeypatch):
        monkeypatch.setenv("JANUS_SEARCH__RERANK__TYPO_KEY", "x")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            warn_unknown_env_once()  # 1st time → warns
            caught_after = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warn_unknown_env_once()  # 2nd time → no warn
                caught_after = caught
            assert all(
                "Unknown JANUS_SEARCH__RERANK__*" not in str(w.message)
                for w in caught_after
            )


# ─── F26: lazy-init race + env binding ────────────────────────────────────


class TestF26LazyInitRace:
    """F26: ``_circuit`` is lazy-initialised; concurrent coroutines
    entering ``_get_circuit`` before init finishes must all see the
    SAME instance. ``_circuit_init_lock`` serialises init.
    """

    @pytest.mark.asyncio
    async def test_f26_single_call_returns_circuit(self):
        cfg = JanusSettings()
        c = await _get_circuit(cfg)
        assert isinstance(c, _RerankCircuit)
        assert c.failure_threshold == cfg.search.rerank.failure_threshold

    @pytest.mark.asyncio
    async def test_f26_concurrent_init_returns_same_instance(self):
        cfg = JanusSettings()

        async def get_c():
            async with _circuit_init_lock:
                return await _get_circuit(cfg)

        results = await asyncio.gather(*[get_c() for _ in range(10)])
        assert all(r is results[0] for r in results), "F26: all concurrent inits must return same singleton"

    def test_f26_env_failure_threshold_loads(self, monkeypatch):
        monkeypatch.setenv("JANUS_SEARCH__RERANK__FAILURE_THRESHOLD", "10")
        monkeypatch.setenv("JANUS_SEARCH__RERANK__RESET_TIMEOUT_SEC", "60.0")
        s = JanusSettings()
        assert s.search.rerank.failure_threshold == 10
        assert s.search.rerank.reset_timeout_sec == 60.0


# ─── F13: module-level session reuse ──────────────────────────────────────


class TestF13SessionReuse:
    """F13: aiohttp.ClientSession is module-level, created lazily.
    Per-call session creation costs 50-100ms TCP/TLS handshake."""

    @pytest.mark.asyncio
    async def test_f13_session_returned_is_same(self):
        s1 = await _get_session()
        s2 = await _get_session()
        assert s1 is s2


# ─── F21: model_name empty ────────────────────────────────────────────────


class TestF21ModelNameEmpty:
    """F21: model_name=\"\" is allowed (llama-server ignores the field
    and uses its --m flag at startup)."""

    def test_f21_empty_model_name_default(self):
        cfg = RerankConfig()
        assert cfg.model_name == ""

    def test_f21_empty_model_name_serialised(self):
        cfg = RerankConfig(model_name="")
        payload = {"model": cfg.model_name or ""}
        assert payload["model"] == ""


# ─── F22: TODO marker present (deferred) ──────────────────────────────────


class TestF22DeferredMarker:
    """F22: /health rerank_circuit_state deferred to v2.1.
    Verify the TODO marker is present in module docstring or code."""

    def test_f22_todo_marker_in_module(self):
        src = inspect.getsource(rerank_mod)
        # Either the F22 note or v2.1 deferred marker must be present
        assert ("F22" in src) or ("v2.1" in src), (
            "F22: TODO marker for deferred /health rerank_circuit_state must be visible in module"
        )


# ─── F23: PicoClaw hook forward-compat ─────────────────────────────────────


class TestF23HookForwardCompat:
    """F23: when PicoClaw hook flips to HTTP /search/memory, this same
    rerank pass must apply (engine facade is the single SoT).
    The 1-liner forward-compat note must be in the module docstring."""

    def test_f23_picoclaw_hook_note_present(self):
        src = inspect.getsource(rerank_mod)
        assert "PicoClaw" in src or "picoclaw" in src
        # Check the note actually references hook + forward-compat
        assert "hook" in src.lower()
        assert "forward-compat" in src.lower() or "forward compat" in src.lower()


# ─── F10: integration site exists ─────────────────────────────────────────


class TestF10IntegrationSite:
    """F10: search_memory calls _maybe_rerank AFTER facts.append loop,
    BEFORE return dict."""

    def test_f10_search_memory_imports_maybe_rerank(self):
        from janus_graph.engine import search_memory as sm
        src = inspect.getsource(sm)
        assert "_maybe_rerank" in src
        # Must be called with the right pattern
        assert "await _maybe_rerank(cfg, query, facts)" in src

    def test_f10_call_between_facts_loop_and_return(self):
        """The await _maybe_rerank call must sit between the facts.append
        loop and the return dict, not outside the try block."""
        from janus_graph.engine import search_memory as sm
        src = inspect.getsource(sm.search_memory)
        lines = src.splitlines()
        # Find the call line
        call_idx = next(
            (i for i, l in enumerate(lines) if "await _maybe_rerank" in l),
            None,
        )
        assert call_idx is not None
        # Find the facts.append line above
        append_idx = next(
            (i for i, l in enumerate(lines) if "facts.append" in l),
            None,
        )
        # Find the return { below
        return_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("return {"))
        )
        # Find the except line below — must NOT be before call
        except_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith("except Exception")),
            None,
        )
        assert append_idx < call_idx < return_idx, "F10 call must sit between facts.append and return"
        assert call_idx < except_idx, "F10 call must be INSIDE the try block"


# ─── F24: concurrent parallel < 10s ───────────────────────────────────────


class TestF24ConcurrentParallel:
    """F24: 5 concurrent _maybe_rerank calls must complete < 10s.
    asyncio.Lock serialises, but each call is ~50-200ms."""

    @pytest.mark.asyncio
    async def test_f24_5_concurrent_complete_under_10s(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = True

        facts = [
            {"fact": f"doc {i}", "name": "", "valid_at": "", "invalid_at": ""}
            for i in range(3)
        ]

        # 100ms delay per call × 5 serialised calls = ~500ms total (well under 10s cap)
        session = _make_fake_session(
            delay=0.1,
            json_payload={
                "results": [
                    {"index": 2, "relevance_score": 5.0},
                    {"index": 0, "relevance_score": 3.0},
                    {"index": 1, "relevance_score": 1.0},
                ]
            },
        )

        facts = [
            {"fact": f"doc {i}", "name": "", "valid_at": "", "invalid_at": ""}
            for i in range(3)
        ]

        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            t0 = time.monotonic()
            results = await asyncio.gather(
                *[_maybe_rerank(cfg, "q", facts) for _ in range(5)]
            )
            elapsed = time.monotonic() - t0
            assert elapsed < 10.0, f"F24: 5 concurrent reranks took {elapsed:.2f}s (cap 10s)"
            # All 5 returned a 3-element reordered list
            assert all(len(r) == 3 for r in results)


# ─── F15: skip malformed (end-to-end via _maybe_rerank) ───────────────────


class TestF15SkipMalformedE2E:
    """F15: malformed entries in /v1/rerank response must be skipped
    (not abort). End-to-end via _maybe_rerank with a mocked response."""

    @pytest.mark.asyncio
    async def test_f15_skip_malformed_response(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = True

        facts = [
            {"fact": f"doc {i}", "name": "", "valid_at": "", "invalid_at": ""}
            for i in range(3)
        ]

        session = _make_fake_session(json_payload={
            "results": [
                {"index": 0, "relevance_score": 5.0},
                {"relevance_score": 9.0},      # F15: missing index
                "bad",                          # F15: not a dict
                {"index": 1},                   # F15: missing score
                {"index": 2, "relevance_score": 1.0},
            ]
        })
        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            result = await _maybe_rerank(cfg, "q", facts)
            assert len(result) == 3
            # Top-ranked valid entry first (index 0)
            assert "doc 0" in result[0]["fact"]


# ─── Integration: circuit breaker end-to-end ──────────────────────────────


class TestIntegrationCircuitEndToEnd:
    """Integration: 3 consecutive failures trip OPEN; next call returns
    original facts unchanged."""

    @pytest.mark.asyncio
    async def test_3_failures_open_circuit_then_degrade(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = True

        facts = [{"fact": f"d{i}", "name": "", "valid_at": "", "invalid_at": ""} for i in range(2)]

        session = _make_fake_session(status=503)
        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            # Failure 1
            r1 = await _maybe_rerank(cfg, "q", facts)
            assert r1 == facts  # degraded
            # Failure 2
            r2 = await _maybe_rerank(cfg, "q", facts)
            assert r2 == facts
            # Failure 3 → trips OPEN
            r3 = await _maybe_rerank(cfg, "q", facts)
            assert r3 == facts

            # Verify circuit is OPEN — 4th call must NOT hit the server
            session.post = MagicMock(side_effect=AssertionError("circuit OPEN must skip HTTP"))
            r4 = await _maybe_rerank(cfg, "q", facts)
            assert r4 == facts


# ─── Integration: full happy-path with mocked aiohttp ─────────────────────


class TestIntegrationHappyPath:
    """Integration: enabled + healthy server → facts reordered by score."""

    @pytest.mark.asyncio
    async def test_full_happy_path_reorders_by_relevance(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = True
        cfg.search.rerank.base_url = "http://127.0.0.1:9999"

        facts = [
            {"fact": "low",   "name": "", "valid_at": "", "invalid_at": ""},
            {"fact": "high",  "name": "", "valid_at": "", "invalid_at": ""},
            {"fact": "mid",   "name": "", "valid_at": "", "invalid_at": ""},
        ]

        # high(1) > mid(2) > low(0) by relevance
        session = _make_fake_session(json_payload={
            "results": [
                {"index": 1, "relevance_score": 9.0},
                {"index": 2, "relevance_score": 5.0},
                {"index": 0, "relevance_score": 1.0},
            ]
        })
        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            result = await _maybe_rerank(cfg, "test query", facts)
            assert [r["fact"] for r in result] == ["high", "mid", "low"]


# ─── Integration: disabled flag is no-op ──────────────────────────────────


class TestIntegrationDisabled:
    """Integration: rerank.enabled=False → return facts unchanged,
    no HTTP call, no circuit state change."""

    @pytest.mark.asyncio
    async def test_disabled_returns_facts_unchanged(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = False

        facts = [{"fact": f"d{i}", "name": "", "valid_at": "", "invalid_at": ""} for i in range(3)]

        session = MagicMock()
        session.closed = False
        session.post = MagicMock(
            side_effect=AssertionError("disabled rerank must not call /v1/rerank")
        )
        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            result = await _maybe_rerank(cfg, "q", facts)
            assert result == facts


# ─── Integration: timeout path degrades gracefully ────────────────────────


class TestIntegrationTimeoutDegrade:
    """Integration: HTTP timeout → record_failure, return facts unchanged.
    No exception bubbles up."""

    @pytest.mark.asyncio
    async def test_timeout_degrades_to_original_order(self):
        cfg = JanusSettings()
        cfg.search.rerank.enabled = True
        cfg.search.rerank.max_candidates = 1  # forces derived timeout = floor=1.0s

        facts = [{"fact": f"d{i}", "name": "", "valid_at": "", "invalid_at": ""} for i in range(3)]

        # 10s delay → well past the 1s timeout → asyncio.TimeoutError
        session = _make_fake_session(delay=10.0)
        with patch.object(rerank_mod, "_get_session", AsyncMock(return_value=session)):
            result = await _maybe_rerank(cfg, "q", facts)
            assert result == facts  # degraded, no exception bubbled
