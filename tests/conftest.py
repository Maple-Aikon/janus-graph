"""Pytest configuration and shared fixtures."""

import os
import tempfile
import pytest
from pathlib import Path
from janus_graph.config import JanusSettings, load_config
from janus_graph.pipeline.queue import EpisodeQueue


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
