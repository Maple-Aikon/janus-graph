"""Baseline scaffolding and configuration tests."""

import re
import socket
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


def _isolate_falkordb_probe(monkeypatch):
    """Neutralise the ONE network probe FalkorDB's constructor performs.

    Why this exists
    ---------------
    ``create_graphiti_instance`` builds a ``FalkorDriver``, whose constructor
    instantiates ``falkordb.asyncio.FalkorDB``. That constructor calls
    ``Is_Cluster(conn)``, which opens a *synchronous* redis connection and
    issues ``INFO server`` just to ask "is this a cluster?" (see
    ``falkordb/asyncio/cluster.py``). So merely *constructing* the engine
    performs real network I/O, and on a host with no FalkorDB it raises
    ``redis.exceptions.ConnectionError``.

    That made this test depend on the machine it ran on: green on Maple's box
    (a live FalkorDB on :6379), red on a GitHub runner (nothing listening).
    The pipeline had never been green -- 40/40 historical runs failed here.

    The test asserts WIRING (which client classes are composed), not data,
    so it does not need a database. ``Is_Cluster`` is patched to answer
    "no" without a socket: the driver is then built exactly as production
    builds it, minus the probe.

    This is a test-side seam only. ``create_graphiti_instance`` itself is
    unchanged, so the daemon/worker/MCP paths keep their real behaviour
    (including the fail-soft handling of an unreachable engine).

    Note the patch target is the symbol *as imported by* ``falkordb.asyncio.falkordb``
    -- patching ``falkordb.asyncio.cluster.Is_Cluster`` would NOT work,
    because the constructor already holds a direct reference to its own
    module-level import.
    """
    from falkordb.asyncio import falkordb as falkordb_asyncio

    monkeypatch.setattr(falkordb_asyncio, "Is_Cluster", lambda conn: False)


def test_create_graphiti_instance_structure(monkeypatch):
    """The composed engine must use the repair client and the embed cache.

    Runs with NO FalkorDB reachable: the constructor's cluster probe is the
    only network call, and it is stubbed (see ``_isolate_falkordb_probe``).
    The port below is a closed one on purpose -- if the probe ever comes
    back, this test fails loudly instead of quietly needing a live server.
    """
    from graphiti_core import Graphiti

    from janus_graph.cache.embed_cache import EmbedCache
    from janus_graph.core.instance import create_graphiti_instance
    from janus_graph.heuristics.repairing_client import SchemaRepairingLLMClient

    _isolate_falkordb_probe(monkeypatch)

    # A port nothing is listening on. Bound and closed so the number is
    # genuinely unused at the moment of the test.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

    cfg = load_config()
    cfg.engine.port = closed_port
    inst = create_graphiti_instance(cfg)
    assert isinstance(inst, Graphiti)
    assert isinstance(inst.llm_client, SchemaRepairingLLMClient)
    assert isinstance(inst.embedder, EmbedCache)

    # The driver must be pointed at the configured endpoint -- proving the
    # stub did not detach the engine from the settings.
    kwargs = inst.driver.client.connection.connection_pool.connection_kwargs
    assert kwargs["host"] == cfg.engine.host
    assert kwargs["port"] == closed_port
