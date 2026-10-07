"""Pytest configuration and shared fixtures.

``_isolated_operator_env`` is autouse and session-wide: ``JanusSettings()`` is
a ``BaseSettings`` whose YAML source is resolved by
``config._resolve_default_yaml_file()`` in this order --
``JANUS_CONFIG_PATH`` -> ``$JANUS_GRAPH_HOME/config.yaml`` -> ``./config.yaml``
-> ``./config.example.yaml``. The daemon exports ``JANUS_GRAPH_HOME``, so a
bare ``JanusSettings()`` inside pytest inherits the operator's LIVE config.
That makes repo-default assertions (``mmr_lambda == 0.5``, ``reranker ==
"mmr"``) fail on the daemon host while passing everywhere else -- 12 tests, all
env-dependent, zero code defects.

Fix mirrors the pattern already used by ``test_config_home_paths.py`` and
``test_queue_path_anchoring.py`` (both pin the same two vars in their own
autouse fixtures); this moves it to session scope so every module inherits it.
Tests that genuinely need a specific value re-pin it with ``monkeypatch``.
"""

import os
import tempfile
from pathlib import Path

import pytest

from janus_graph.config import JanusSettings, load_config
from janus_graph.pipeline.queue import EpisodeQueue

# Vars that let pytest escape into the operator's real installation.
_OPERATOR_ENV_KEYS = (
    "JANUS_CONFIG_PATH",
    "JANUS_GRAPH_HOME",
)


@pytest.fixture(autouse=True)
def _isolated_operator_env(tmp_path_factory, monkeypatch):
    """Stop tests from reading the operator's live config.yaml.

    ``JANUS_CONFIG_PATH`` points at a non-existent file and
    ``JANUS_GRAPH_HOME`` at a temp dir, so ``_resolve_default_yaml_file()``
    finds no operator YAML. ``config.example.yaml`` in the repo root still
    resolves as the final fallback, which is the repo's committed default --
    exactly what these tests assert.
    """
    tmp = tmp_path_factory.mktemp("janus-home")
    monkeypatch.setenv("JANUS_CONFIG_PATH", str(tmp / "does-not-exist.yaml"))
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(tmp))
    yield
    for k in _OPERATOR_ENV_KEYS:
        os.environ.pop(k, None)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_queue(temp_dir):
    db_path = temp_dir / "test_episodes.db"
    return EpisodeQueue(str(db_path))


@pytest.fixture
def test_settings(temp_dir):
    settings = JanusSettings()
    settings.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    settings.heuristics.quirks_log_path = str(temp_dir / "quirks.jsonl")
    settings.report.sinks.file.path = str(temp_dir / "report.jsonl")
    return settings
