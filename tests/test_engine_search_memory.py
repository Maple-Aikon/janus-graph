"""Tests for ``janus_graph.engine.search_memory`` facade.

v0.5.0 (PR 1 of plan ``add-http-searchmemory-endpoint-via-shared-engine-module-v2-...``):

  - Test A: graphiti-core version pin drift-check (SOFT WARNING, not hard
    fail -- see 2026-09-16 lesson). Plan section 5 (F7) originally pinned
    ``>=0.20.0,<0.21.0a0`` because the v0.4.7 hotfix rationale depends on
    ``Graphiti.search()`` NOT accepting ``search_config`` (it hardcodes
    ``EDGE_HYBRID_SEARCH_RRF``); only the module-level ``graphiti_search``
    does. PEP 440: ``<0.21.0`` does NOT exclude pre-releases like
    ``0.21.0a1`` / ``0.21.0rc1`` because they sort below ``0.21.0`` on
    the version line.

    Why SOFT (2026-09-16): production runs ``graphiti-core`` 0.30.1.
    Live re-validation (em 2026-09-16, see plan F7 follow-up note): at
    0.30.1, ``graphiti_search`` still accepts ``config: SearchConfig`` and
    ``Graphiti.search`` still does NOT -- so the F7 rationale HOLDS at
    0.30.1. A hard ``assert major_minor == (0, 20)`` would break the test
    suite in production, forcing a pyproject.toml downgrade that would
    actually be unsafe (0.30.1 has fixes we depend on).

    UPDATED 2026-10-10: the pin is now DECLARED exactly (``==0.30.1``) and
    the test compares the installed version against that declared pin
    rather than a hardcoded ``0.20.*`` range. The old range check fired on
    every single run, which is noise that teaches people to ignore the one
    warning this test exists to raise.

  - Test B: NFC/NFD diacritic normalization for stacked diacritics.

  - Test B: NFC/NFD diacritic normalization for stacked diacritics.
    Plan §1.3 (F1): Vietnamese names with stacked diacritics (e.g.
    ``"ệ"`` = base + circumflex + dot-below) can be encoded as either
    NFC (1 codepoint ``\u1ecb``) or NFD (3 codepoints: ``e`` + ``\u0302``
    + ``\u0323``). Naive codepoint-scanning detection breaks under the
    second form. The engine must normalize to NFC before running the
    dual-query (canonical form + stripped form) hook-level policy.

  - Test C: facade smoke test — engine call delegates to caller-provided
    graphiti singleton (MCP path passes its closure-local one; HTTP path
    will pass the daemon's ``ctx.graphiti_instance`` in PR 2). Verifies
    no accidental lazy-init fires when ``graphiti`` is provided.

  - Test D: facade error contract — exception from graphiti becomes
    ``{"success": False, "error": ..., "group_id": ..., "query": ...}``
    so the MCP wire contract is preserved.
"""

from __future__ import annotations

import tomllib
import unicodedata
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from janus_graph.config import JanusSettings
from janus_graph.engine.search_memory import search_memory

# ---------------------------------------------------------------------------
# Test A — graphiti-core version pin drift-check
# ---------------------------------------------------------------------------


def test_graphiti_core_pin_drift_check():
    """Guard that the INSTALLED ``graphiti-core`` matches the DECLARED pin.

    History: plan section 5 (F7) originally pinned ``>=0.20.0,<0.21.0a0``
    because the v0.4.7 hotfix rationale depends on ``Graphiti.search()`` NOT
    accepting ``search_config`` (it hardcodes ``EDGE_HYBRID_SEARCH_RRF``);
    only the module-level ``graphiti_search`` does. PEP 440: ``<0.21.0`` does
    NOT exclude pre-releases like ``0.21.0a1`` / ``0.21.0rc1`` because they
    sort below ``0.21.0`` on the version line.

    2026-10-10: production ran 0.30.1 while ``pyproject.toml`` declared only
    ``>=0.20`` with no upper bound, so the de-facto version was UNDECLARED.
    The pin is now exact: ``==0.30.1``.

    This test therefore compares the installed version against the pin read
    from ``pyproject.toml`` instead of hardcoding a historical range. Keeping
    the hardcoded ``(0, 20)`` check would make this warn on EVERY run -- noise
    that trains people to ignore the one warning the test exists to raise.

    SOFT WARNING (2026-09-16): we do not hard-fail. An exact-pin mismatch
    should surface as an explicit re-validation decision (F7 re-check at the
    new version), not as an opaque suite break.
    """
    try:
        graphiti_ver_str = version("graphiti-core")
    except PackageNotFoundError:
        pytest.skip("graphiti-core not installed in this env")

    declared = None
    try:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with open(pyproject, "rb") as fh:
            deps = tomllib.load(fh)["project"]["dependencies"]
        for spec in deps:
            if spec.replace(" ", "").startswith("graphiti-core"):
                declared = spec
                break
    except Exception:  # noqa: BLE001 - a drift guard must never break the suite
        declared = None

    if declared is None:
        pytest.skip("could not read the graphiti-core pin from pyproject.toml")

    if "==" in declared:
        pinned = declared.split("==", 1)[1].strip()
        if graphiti_ver_str != pinned:
            warnings.warn(
                "graphiti-core=%r does NOT match the declared pin %r in "
                "pyproject.toml. The F7 hotfix rationale (module-level "
                "graphiti_search accepts `config`; Graphiti.search does not) "
                "was validated at 0.30.1 on 2026-09-16 and holds at this "
                "version. Before bumping the pin, re-validate F7 and re-run "
                "the search gold-set harness. See plan F7 follow-up note."
                % (graphiti_ver_str, pinned),
                UserWarning,
                stacklevel=2,
            )
    else:
        # Range / unpinned specifier: keep the historical drift signal.
        parts = graphiti_ver_str.split(".")
        major_minor = (int(parts[0]), int(parts[1]))
        if major_minor != (0, 20):
            warnings.warn(
                "graphiti-core=%r drifted outside the plan F7 pin range 0.20.*, "
                "and pyproject.toml declares %r (not an exact pin). The v0.4.7 "
                "hotfix rationale (module-level graphiti_search vs "
                "Graphiti.search with no search_config kwarg) was validated "
                "against 0.20.x. Em 2026-09-16 live re-validation confirmed F7 "
                "still holds at 0.30.1. Prefer an exact pin." % (graphiti_ver_str, declared),
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Test B — NFC/NFD diacritic normalization
# ---------------------------------------------------------------------------


def _has_diacritics(s: str) -> bool:
    """Return True if ``s`` has any combining marks or precomposed diacritics."""
    for ch in s:
        # Combining marks (NFD decomposition product): category 'M'
        if unicodedata.category(ch) == "Mn":
            return True
        # Precomposed diacritic letters (Latin Extended A/B, etc.)
        if unicodedata.name(ch, "").startswith(("LATIN ", "LATIN ")) and (
            "WITH " in unicodedata.name(ch, "") or "COMBINING" in unicodedata.name(ch, "")
        ):
            return True
    return False


@pytest.mark.parametrize(
    "name_codepoints",
    [
        # NFC form — single codepoint (precomposed)
        "Thúy Vi",  # ú = U+00FA (precomposed)
        "Thuy\u0309 Vi",  # explicit NFC: u + combining acute = same as above
        # NFD form — decomposed; the combining acute is its own codepoint
        "Thuy\u0309 Vi",  # already NFC above; this becomes the decomposed variant
        # Stacked diacritics — the F1 edge case
        "Hi\u1ecb" + "ên",  # NFC: ệ = U+1EC3 (base + circumflex + dot-below precomposed)
        "Hi\u1ecb" + "ên",  # same NFC form
        "Hi\u0065\u0302\u0323n",  # NFD: e + U+0302 (combining circumflex) + U+0323 (combining dot below)
    ],
)
def test_diacritic_detection_normalizes_form(name_codepoints):
    """F1 detection must normalize to NFC before scanning.

    Plan §1.3 (fix C applied 2026-09-16): Vietnamese stacked diacritics
    like ``"ệ"`` (e + circumflex + dot-below) can be encoded as either:

      - NFC: U+1EC3 ``ệ`` (single precomposed codepoint)
      - NFD: ``e`` (U+0065) + U+0302 (combining circumflex accent)
                              + U+0323 (combining dot below)

    A naive codepoint scan like ``any(ord(c) > 127 for c in s)`` would
    miss the NFD form's combining marks because combining marks sit in
    the U+0300..U+036F range, BELOW 128 in the BMP for some entries
    (e.g. U+0323). The engine's diacritic detection (used by the hook
    dual-query policy in PR 2) MUST call
    ``unicodedata.normalize("NFC", s)`` before scanning.

    This test asserts that AFTER NFC normalization the input always
    contains at least one precomposed diacritic letter, regardless of
    whether the original input was NFC or NFD.
    """
    normalized = unicodedata.normalize("NFC", name_codepoints)
    assert _has_diacritics(normalized), (
        f"NFC-normalized {normalized!r} (from input "
        f"{name_codepoints!r}) lost its diacritic signal. F1 "
        f"detection is broken — the hook dual-query will silently "
        f"skip VN facts."
    )


def test_nfc_nfd_round_trip_preserves_detection():
    """Round-trip: NFC → NFD → NFC must yield the same detection result.

    Plan §1.3 (fix C): ensures the diacritic detection is invariant
    under Unicode normalization, not just one-way.
    """
    original = "Thúy Vi"  # NFC
    decomposed = unicodedata.normalize("NFD", original)
    re_composed = unicodedata.normalize("NFC", decomposed)

    assert _has_diacritics(re_composed)
    assert _has_diacritics(unicodedata.normalize("NFC", decomposed))
    # The detection must agree on both forms
    assert _has_diacritics(original) == _has_diacritics(re_composed)


# ---------------------------------------------------------------------------
# Test C — facade smoke test (caller-provided graphiti)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_settings() -> JanusSettings:
    """Minimal JanusSettings for engine smoke tests."""
    return JanusSettings()


@pytest.mark.asyncio
async def test_engine_facade_uses_caller_graphiti(base_settings):
    """When caller passes ``graphiti``, engine MUST NOT lazy-init.

    The MCP path passes its closure-local singleton (to avoid duplicate
    graphiti instances per tool call). The future HTTP path will pass
    the daemon's ctx.graphiti_instance. If the engine ignored the
    passed singleton and lazy-init'd its own, every HTTP call would
    spin up a fresh graphiti (with EmbedCache, SchemaRepairingLLMClient,
    etc.) — a perf disaster.

    We mock graphiti + graphiti_search to verify delegation without
    needing a live FalkorDB.
    """
    mock_graphiti = MagicMock()
    mock_graphiti.clients = MagicMock()

    # Patch the module-level graphiti_search the engine imports
    from janus_graph.engine import search_memory as engine_module

    mock_graphiti_search = AsyncMock(
        return_value=MagicMock(edges=[]),  # empty results
    )

    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_graphiti_search
    try:
        result = await search_memory(
            base_settings,
            "test query",
            limit=5,
            graphiti=mock_graphiti,
        )
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is True
    assert result["query"] == "test query"
    assert result["count"] == 0
    assert result["results"] == []
    # Verify graphiti_search was called exactly once with caller's clients
    assert mock_graphiti_search.await_count == 1
    call_kwargs = mock_graphiti_search.await_args.kwargs
    assert call_kwargs["clients"] is mock_graphiti.clients
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["group_ids"] == [base_settings.graphiti.group_id]


# ---------------------------------------------------------------------------
# Test D — facade error contract (preserves MCP wire shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_facade_error_returns_mcp_wire_shape(base_settings):
    """Exception inside engine becomes ``{success: False, error, ...}``.

    Preserves the MCP ``search_memory`` error contract so the MCP tool's
    ``try/except`` is now redundant (engine handles it). PR 2 HTTP path
    relies on the same shape.
    """
    mock_graphiti = MagicMock()
    mock_graphiti.clients = MagicMock()

    from janus_graph.engine import search_memory as engine_module

    mock_graphiti_search = AsyncMock(
        side_effect=RuntimeError("falkordb unreachable"),
    )

    original = engine_module.graphiti_search
    engine_module.graphiti_search = mock_graphiti_search
    try:
        result = await search_memory(
            base_settings,
            "test query",
            limit=5,
            graphiti=mock_graphiti,
        )
    finally:
        engine_module.graphiti_search = original

    assert result["success"] is False
    assert "falkordb unreachable" in result["error"]
    assert result["query"] == "test query"
    assert result["group_id"] == base_settings.graphiti.group_id


# ---------------------------------------------------------------------------
# Test E — reranker recipe selection (2026-10-04)
# ---------------------------------------------------------------------------
#
# Before this the recipe was hardcoded to EDGE_HYBRID_SEARCH_MMR at
# search_memory.py. These tests pin the config-driven selection, and — more
# importantly — pin the FAIL-CLOSED behaviour: a typo must never reorder or
# empty search results, it must land on the pre-existing MMR recipe.


async def _captured_recipe(monkeypatch, settings):
    """Run the facade with graphiti_search mocked; return the SearchConfig
    that was actually handed to graphiti_core."""
    from janus_graph.engine import search_memory as engine_module

    captured = {}

    async def _fake_search(**kwargs):
        captured["config"] = kwargs.get("config")
        return MagicMock(edges=[])

    monkeypatch.setattr(engine_module, "graphiti_search", _fake_search)

    mock_graphiti = MagicMock()
    mock_graphiti.clients = MagicMock()

    async def _noop(*a, **k):
        return None

    mock_graphiti.clients.embedder.create = _noop

    result = await search_memory(
        settings, "reranker selection probe", limit=5, graphiti=mock_graphiti
    )
    assert result["success"] is True, result
    return captured["config"]


@pytest.mark.asyncio
async def test_default_config_keeps_mmr_recipe(monkeypatch):
    """No config change => byte-identical behaviour to pre-2026-10-04."""
    settings = JanusSettings()
    assert settings.search.reranker == "mmr"
    cfg = await _captured_recipe(monkeypatch, settings)
    from graphiti_core.search.search_config import EdgeReranker

    assert cfg.edge_config.reranker == EdgeReranker.mmr


@pytest.mark.asyncio
async def test_reranker_rrf_selects_rrf_recipe(monkeypatch):
    settings = JanusSettings()
    settings.search.reranker = "rrf"
    cfg = await _captured_recipe(monkeypatch, settings)
    from graphiti_core.search.search_config import EdgeReranker

    assert cfg.edge_config.reranker == EdgeReranker.rrf


@pytest.mark.asyncio
async def test_reranker_alias_resolves(monkeypatch):
    """graphiti's own enum value must work, so an operator copying it from
    a stack trace gets what they asked for."""
    settings = JanusSettings()
    settings.search.reranker = "reciprocal_rank_fusion"
    cfg = await _captured_recipe(monkeypatch, settings)
    from graphiti_core.search.search_config import EdgeReranker

    assert cfg.edge_config.reranker == EdgeReranker.rrf


@pytest.mark.asyncio
async def test_unknown_reranker_fails_closed_to_mmr(monkeypatch):
    """A junk value must NOT raise, and must NOT silently pick a recipe.

    pydantic only runs field validators on assignment when
    validate_assignment=True (SearchConfig does not set it), so this is the
    exact path that would have KeyError'd the whole search.
    """
    settings = JanusSettings()
    settings.search.reranker = "definitely-not-a-reranker"
    cfg = await _captured_recipe(monkeypatch, settings)
    from graphiti_core.search.search_config import EdgeReranker

    assert cfg.edge_config.reranker == EdgeReranker.mmr


@pytest.mark.asyncio
async def test_reranker_floor_kept_on_config(monkeypatch):
    """rrf() honours min_score, so the -2.0 floor must still be pushed down.

    rrf() filters ``score >= min_score`` and its scores are
    1/(rank+1) sums — always positive — so -2.0 never binds, but the
    floor is also what protects a future non-positive reranker. Guard the
    plumbing, not the value.
    """
    settings = JanusSettings()
    settings.search.reranker = "rrf"
    cfg = await _captured_recipe(monkeypatch, settings)
    assert cfg.reranker_min_score == pytest.approx(-2.0)
