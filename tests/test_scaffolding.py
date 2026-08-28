"""Baseline scaffolding and configuration tests."""

from pathlib import Path
from janus_graph import __version__
from janus_graph.config import load_config, JanusSettings


def test_version():
    assert __version__ == "0.1.0"


def test_config_load_defaults():
    cfg = load_config()
    assert isinstance(cfg, JanusSettings)
    assert cfg.engine.port == 6379
    assert cfg.pipeline.worker_concurrency == 6
    assert cfg.heuristics.auto_repair is True
    assert "edge_duplicate" in cfg.heuristics.active_rules


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("JANUS_ENGINE__PORT", "6380")
    monkeypatch.setenv("JANUS_PIPELINE__WORKER_CONCURRENCY", "12")
    cfg = load_config()
    assert cfg.engine.port == 6380
    assert cfg.pipeline.worker_concurrency == 12
