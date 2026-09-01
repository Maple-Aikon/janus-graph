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
    resolve_search_params,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Strip JANUS_SEARCH__* env vars before each test."""
    for k in ("JANUS_SEARCH__SIM_MIN_SCORE", "JANUS_SEARCH__MMR_LAMBDA"):
        os.environ.pop(k, None)
    yield
    for k in ("JANUS_SEARCH__SIM_MIN_SCORE", "JANUS_SEARCH__MMR_LAMBDA"):
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
