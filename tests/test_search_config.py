"""Tests for janus_graph.config.SearchConfig and resolve_search_params.

Covers the env > config > default precedence and the fail-soft invalid-env
fallback that prevents ``MCP restart + bad env`` from brick-ing the
search_memory tool.

v0.4.6: introduced alongside ``SearchConfig`` so operators can tune the
graphiti-core SearchConfig without rebuilding the wheel.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from janus_graph.config import (
    JanusSettings,
    SearchConfig,
    resolve_reranker_min_score,
    resolve_search_params,
)

_SEARCH_ENV_KEYS = (
    "JANUS_SEARCH__SIM_MIN_SCORE",
    "JANUS_SEARCH__MMR_LAMBDA",
    "JANUS_SEARCH__RERANKER_MIN_SCORE",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Strip JANUS_SEARCH__* env vars before each test."""
    for k in _SEARCH_ENV_KEYS:
        os.environ.pop(k, None)
    yield
    for k in _SEARCH_ENV_KEYS:
        os.environ.pop(k, None)


def test_search_config_defaults():
    """SearchConfig has graphiti-core aligned defaults (0.6 / 0.5)."""
    sc = SearchConfig()
    assert sc.sim_min_score == pytest.approx(0.6)
    assert sc.mmr_lambda == pytest.approx(0.5)


def test_janus_settings_has_search_field():
    """The Settings root wires SearchConfig under .search."""
    s = JanusSettings()
    assert isinstance(s.search, SearchConfig)
    assert s.search.sim_min_score == pytest.approx(0.6)
    assert s.search.mmr_lambda == pytest.approx(0.5)


def test_resolve_search_params_defaults():
    """No env → cfg.search defaults 0.6 / 0.5."""
    s = JanusSettings()
    sim, mmr = resolve_search_params(s)
    assert sim == pytest.approx(0.6)
    assert mmr == pytest.approx(0.5)


def test_resolve_search_params_env_overrides_defaults():
    """Env set with valid value beats defaults."""
    os.environ["JANUS_SEARCH__SIM_MIN_SCORE"] = "0.45"
    os.environ["JANUS_SEARCH__MMR_LAMBDA"] = "0.7"
    s = JanusSettings()
    sim, mmr = resolve_search_params(s)
    assert sim == pytest.approx(0.45)
    assert mmr == pytest.approx(0.7)


def test_resolve_search_params_yaml_overrides_default():
    """YAML config overrides built-in defaults (env unset)."""
    cfg_dict = {"search": {"sim_min_score": 0.55, "mmr_lambda": 0.65}}
    s = JanusSettings(**cfg_dict)
    sim, mmr = resolve_search_params(s)
    assert sim == pytest.approx(0.55)
    assert mmr == pytest.approx(0.65)


def test_resolve_search_params_env_beats_yaml():
    """Env beats YAML (canonical precedence)."""
    os.environ["JANUS_SEARCH__SIM_MIN_SCORE"] = "0.40"
    cfg_dict = {"search": {"sim_min_score": 0.55, "mmr_lambda": 0.65}}
    s = JanusSettings(**cfg_dict)
    sim, mmr = resolve_search_params(s)
    assert sim == pytest.approx(0.40)  # env wins
    assert mmr == pytest.approx(0.65)  # YAML wins over default


@pytest.mark.parametrize(
    "bad_value,reason",
    [
        ("abc", "non-numeric"),
        ("NaN", "NaN"),
        ("nan", "NaN lowercase"),
        ("1.5", "above 1.0"),
        ("-0.3", "negative"),
        ("", "empty"),
        ("   ", "whitespace"),
    ],
)
def test_resolve_search_params_fail_soft_invalid_env(bad_value, reason):
    """Invalid env falls back to cfg.search default without crashing.

    Note: ``JanusSettings()`` is constructed BEFORE the bad env is set,
    because pydantic-settings raises ``ValidationError`` at instantiation
    time. ``resolve_search_params`` re-reads ``os.environ`` directly so the
    helper is what enforces the fail-soft path.
    """
    s = JanusSettings()  # built from clean env → defaults
    os.environ["JANUS_SEARCH__SIM_MIN_SCORE"] = bad_value
    sim, mmr = resolve_search_params(s)
    # Should NOT crash; should fall back to 0.6 default
    assert sim == pytest.approx(0.6), f"failed for {reason}: got {sim}"
    assert mmr == pytest.approx(0.5)


def test_resolve_search_params_boundary_valid():
    """Edge-of-range values (0.0 and 1.0) are accepted."""
    os.environ["JANUS_SEARCH__SIM_MIN_SCORE"] = "0.0"
    os.environ["JANUS_SEARCH__MMR_LAMBDA"] = "1.0"
    s = JanusSettings()
    sim, mmr = resolve_search_params(s)
    assert sim == pytest.approx(0.0)
    assert mmr == pytest.approx(1.0)


def test_resolve_search_params_logs_warning_on_invalid(caplog):
    """Fail-soft path emits a warning log with parameter name."""
    import logging
    caplog.set_level(logging.WARNING, logger="janus_graph.config")
    s = JanusSettings()  # clean env first
    os.environ["JANUS_SEARCH__SIM_MIN_SCORE"] = "garbage"
    with caplog.at_level(logging.WARNING):
        sim, _ = resolve_search_params(s)
    assert sim == pytest.approx(0.6)
    # Should have logged a warning naming the param
    assert any("sim_min_score" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# v0.4.8 — reranker_min_score (the MMR-silently-returns-nothing bug)
# ---------------------------------------------------------------------------


def test_search_config_reranker_min_score_default_is_negative():
    """Default floor must be negative.

    graphiti_core ships 0.0. MMR scores are
    ``mmr_lambda*cos + (mmr_lambda-1)*max_sim``, which is negative for any
    redundant candidate whenever mmr_lambda < 1, so a 0.0 floor filters
    out essentially the whole result set. graphiti_core's own
    ``maximal_marginal_relevance()`` defaults to -2.0 for the same reason.
    """
    sc = SearchConfig()
    assert sc.reranker_min_score < 0, (
        "a non-negative reranker_min_score re-arms the MMR empty-result bug"
    )
    assert sc.reranker_min_score == pytest.approx(-2.0)


def test_resolve_reranker_min_score_default():
    """No env → cfg.search default (-2.0)."""
    assert resolve_reranker_min_score(JanusSettings()) == pytest.approx(-2.0)


def test_resolve_reranker_min_score_yaml_overrides_default():
    """A negative YAML value is honoured (config beats builtin default)."""
    s = JanusSettings()
    s.search.reranker_min_score = -1.5
    assert resolve_reranker_min_score(s) == pytest.approx(-1.5)


def test_resolve_reranker_min_score_env_beats_yaml():
    """env > config."""
    s = JanusSettings()
    s.search.reranker_min_score = -1.5
    os.environ["JANUS_SEARCH__RERANKER_MIN_SCORE"] = "-0.25"
    assert resolve_reranker_min_score(s) == pytest.approx(-0.25)


def test_resolve_reranker_min_score_accepts_full_signed_range():
    """Negative values are legal — unlike sim_min_score/mmr_lambda [0,1]."""
    for raw, expected in (("-2.0", -2.0), ("-0.001", -0.001), ("1.0", 1.0)):
        os.environ["JANUS_SEARCH__RERANKER_MIN_SCORE"] = raw
        assert resolve_reranker_min_score(JanusSettings()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad",
    ["-3.5", "5.0", "abc", "NaN", ""],
    ids=["below_range", "above_range", "not_numeric", "nan", "blank"],
)
def test_resolve_reranker_min_score_fail_soft(bad):
    """Invalid env values never raise and never return a bad number.

    Note: ``JanusSettings()`` is constructed BEFORE the bad env is set,
    because pydantic-settings raises ``ValidationError`` at instantiation
    time (this is true of the pre-existing sim_min_score / mmr_lambda
    fields too, not something introduced here).
    ``resolve_reranker_min_score`` re-reads ``os.environ`` directly, so
    the helper is what enforces the fail-soft path.
    """
    s = JanusSettings()
    os.environ["JANUS_SEARCH__RERANKER_MIN_SCORE"] = bad
    assert resolve_reranker_min_score(s) == pytest.approx(-2.0)


def test_resolve_reranker_min_score_warns_on_zero(caplog):
    """0.0 is in range but is the bug re-armed → warn loudly, still honour it."""
    import logging

    s = JanusSettings()
    caplog.set_level(logging.WARNING, logger="janus_graph.config")
    os.environ["JANUS_SEARCH__RERANKER_MIN_SCORE"] = "0"
    with caplog.at_level(logging.WARNING):
        got = resolve_reranker_min_score(s)
    assert got == pytest.approx(0.0), "0 is legal; the resolver must not silently override it"
    assert any("reranker_min_score" in r.message for r in caplog.records)


def test_resolve_reranker_min_score_is_separate_from_search_params():
    """resolve_search_params keeps its 2-tuple contract.

    Widening it to a 3-tuple would silently break every existing
    ``sim, mmr = resolve_search_params(cfg)`` unpack site.
    """
    s = JanusSettings()
    out = resolve_search_params(s)
    assert isinstance(out, tuple) and len(out) == 2


def test_search_memory_pushes_floor_onto_search_config():
    """Regression guard for the actual bug.

    search_memory must assign the resolved floor onto the *top-level*
    SearchConfig. graphiti_core's search.py:185-222 reads
    ``config.reranker_min_score`` (not the per-entity sub-configs) and
    forwards it straight into the reranker, so pushing sim_min_score /
    mmr_lambda onto the sub-configs alone leaves the floor at 0 and MMR
    returns nothing.
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getfile(JanusSettings)).parent / "engine" / "search_memory.py"
    tree = ast.parse(src.read_text())

    assigned = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "reranker_min_score"
        and isinstance(node.ctx, ast.Store)
    ]
    targets = {ast.unparse(a) for a in assigned}
    assert "search_config.reranker_min_score" in targets, (
        f"search_memory never assigns the floor; found {targets or 'nothing'}"
    )
