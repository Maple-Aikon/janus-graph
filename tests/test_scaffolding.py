"""Baseline scaffolding and configuration tests."""

import re
import tomllib
from pathlib import Path

from janus_graph import VERSION_UNRESOLVED, __version__, __version_source__
from janus_graph.config import JanusSettings, load_config

# PEP 440-ish. Deliberately does NOT match VERSION_UNRESOLVED ("0.0.0+unresolved")
# so a broken SoT resolution fails loudly instead of reporting a plausible number.
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_version():
    """Version must be derived from pyproject, never a literal of its own."""
    assert __version__ != VERSION_UNRESOLVED, "pyproject.toml version could not be resolved"
    assert VERSION_PATTERN.match(__version__), f"not a version: {__version__!r}"


def test_version_has_a_single_source():
    """The invariant that caused this refactor: ONE literal, in pyproject.

    Guards against someone 'fixing' a version by adding a literal to a report
    site, which is how 0.1.0 / 0.5.0 / v0.7.1 all coexisted.
    """
    assert Path(__version_source__).name == "pyproject.toml"
    with open(__version_source__, "rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]
    assert __version__ == pyproject_version


def test_version_report_sites_agree():
    """Every surface that reports a version must report the same one.

    The three sites are what disagreed before (0.1.0 / 0.5.0 / stale metadata).
    """
    from janus_graph.daemon.http_server import __version__ as http_version

    assert http_version == __version__
    assert VERSION_PATTERN.match(http_version)


def test_config_load_defaults():
    cfg = load_config()
    assert isinstance(cfg, JanusSettings)
    assert cfg.engine.port == 6379
    assert cfg.pipeline.worker_concurrency == 2
    assert cfg.heuristics.auto_repair is True
    assert "edge_duplicate" in cfg.heuristics.active_rules


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("JANUS_ENGINE__PORT", "6380")
    monkeypatch.setenv("JANUS_PIPELINE__WORKER_CONCURRENCY", "12")
    cfg = load_config()
    assert cfg.engine.port == 6380
    assert cfg.pipeline.worker_concurrency == 12


def test_create_graphiti_instance_structure():
    from graphiti_core import Graphiti

    from janus_graph.cache.embed_cache import EmbedCache
    from janus_graph.core.instance import create_graphiti_instance
    from janus_graph.heuristics.repairing_client import SchemaRepairingLLMClient

    cfg = load_config()
    inst = create_graphiti_instance(cfg)
    assert isinstance(inst, Graphiti)
    assert isinstance(inst.llm_client, SchemaRepairingLLMClient)
    assert isinstance(inst.embedder, EmbedCache)
