"""CWD-relative path defaults must anchor to JANUS_GRAPH_HOME, not os.getcwd().

Regression guard for 2026-10-02 (Option B hardening). Before the fix,
``EpisodeQueue()`` with no argument resolved its default via
``Path(db_path).resolve()``, which anchors a relative path to the process
CWD: run from ``/tmp`` it produced ``/tmp/data/queue.db`` and silently
created a second, CWD-scoped SQLite database. The same shape was LIVE in
``pipeline/dream.py`` and ``pipeline/cron.py`` whenever the caller passed a
``core.contracts.Settings`` (that dataclass keeps the path under ``paths``,
not ``pipeline``, so the old ``hasattr`` guard took its relative branch).

Every test here runs with a deliberately foreign CWD and asserts the path
landed under the home directory. ``test_bare_episode_queue_does_not_follow_cwd``
is the negative control for the pre-fix behaviour: it fails on the old
``Path(db_path).resolve()`` and passes now.
"""

import os
import sqlite3
from pathlib import Path

import pytest

from janus_graph.config import resolve_home_relative
from janus_graph.core.contracts import Settings
from janus_graph.pipeline.queue import EpisodeQueue

# Deliberately unrelated to the repo. NOTE: pytest --basetemp also lands under
# /tmp, so "not under CWD" is only meaningful for the CWD-relative *file*
# path (<cwd>/data/queue.db), not for the temp home itself.
FOREIGN_CWD = "/tmp"


@pytest.fixture(autouse=True)
def _foreign_cwd_and_home(tmp_path, monkeypatch):
    """Run from a CWD unrelated to the repo, with an isolated home.

    ``monkeypatch.chdir`` restores CWD on teardown, so no other test in the
    suite is affected.
    """
    home = tmp_path / "janus-home"
    home.mkdir()
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(home))
    monkeypatch.chdir(FOREIGN_CWD)
    return home


# ── the helper itself ────────────────────────────────────────────────────


def test_resolve_home_relative_anchors_relative_to_home(_foreign_cwd_and_home):
    resolved = resolve_home_relative("./data/queue.db")
    assert Path(resolved) == _foreign_cwd_and_home / "data" / "queue.db"


def test_resolve_home_relative_leaves_absolute_untouched(_foreign_cwd_and_home, tmp_path):
    pinned = tmp_path / "elsewhere" / "queue.db"
    assert Path(resolve_home_relative(str(pinned))) == pinned


def test_resolve_home_relative_is_not_cwd_relative(_foreign_cwd_and_home):
    """The direct statement of the bug: cwd must not leak into the result."""
    assert not Path(resolve_home_relative("data/queue.db")).is_relative_to(
        Path(os.getcwd()) / "data"
    )


def test_resolve_home_relative_honours_env_home_not_dot_janus(tmp_path, monkeypatch):
    """Precedence: env var wins over the ~/.janus-graph fallback."""
    home = tmp_path / "env-home"
    home.mkdir()
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(home))
    monkeypatch.chdir(FOREIGN_CWD)
    assert Path(resolve_home_relative("./data/queue.db")) == home / "data" / "queue.db"


# ── EpisodeQueue: the actual regression ──────────────────────────────────


def test_bare_episode_queue_does_not_follow_cwd(_foreign_cwd_and_home):
    """Bare ``EpisodeQueue()`` must anchor to home, never to the CWD.

    Negative control: on pre-fix code this asserted ``/tmp/data/queue.db``
    (because ``Path(db_path).resolve()`` anchors to cwd) and the test failed.
    That file was the decoy-database shape this closes.
    """
    queue = EpisodeQueue()

    assert queue.db_path == _foreign_cwd_and_home / "data" / "queue.db"
    assert queue.db_path.is_absolute()


def test_bare_episode_queue_creates_db_under_home_not_cwd(_foreign_cwd_and_home):
    """The visible symptom: file lands under home, and no cwd-relative copy."""
    EpisodeQueue()

    assert (_foreign_cwd_and_home / "data" / "queue.db").exists()
    assert not (Path(FOREIGN_CWD) / "data" / "queue.db").exists()


def test_explicit_absolute_path_still_wins(_foreign_cwd_and_home, tmp_path):
    """An explicit operator-pinned absolute path must not be re-anchored."""
    pinned = tmp_path / "pinned" / "queue.db"
    queue = EpisodeQueue(db_path=str(pinned))
    assert queue.db_path == pinned


def test_explicit_relative_path_is_anchored_to_home(_foreign_cwd_and_home):
    """Relative input is anchored too — the default is not special-cased."""
    queue = EpisodeQueue(db_path="./data/custom.db")
    assert queue.db_path == _foreign_cwd_and_home / "data" / "custom.db"


# ── dream / cron: the branch that was actually live ──────────────────────


def test_contracts_settings_queue_path_is_not_cwd_relative(_foreign_cwd_and_home):
    """documents why dream/cron needed the ``paths`` fallback.

    ``core.contracts.Settings`` exposes ``paths.queue_db_path`` and has NO
    ``pipeline.queue_db_path``, so the removed ``hasattr(cfg.pipeline, ...)``
    guard was False and the ``"./data/queue.db"`` branch ran. This assertion
    is what makes the old branch reachable, not dead code.
    """
    settings = Settings()
    assert not hasattr(settings.pipeline, "queue_db_path")

    resolved = resolve_home_relative(str(settings.paths.queue_db_path))
    assert Path(resolved) == _foreign_cwd_and_home / "data" / "queue.db"


def test_dream_anchors_contracts_settings_queue_path(_foreign_cwd_and_home):
    """The branch that WAS reachable: dream runs to completion on a bare
    ``contracts.Settings`` (cron does not -- it raises on
    ``pipeline.drain_batch_size`` before reaching its db_path line, measured
    2026-10-02). EpisodeQueue is stubbed so the assertion is on *which path
    dream resolved*, not on the work it then does.
    """
    import asyncio

    import janus_graph.pipeline.dream as dream_mod

    captured = {}

    class _CapturingQueue:
        def __init__(self, db_path=None, **_kwargs):
            captured["path"] = str(db_path)

    original = dream_mod.EpisodeQueue
    dream_mod.EpisodeQueue = _CapturingQueue
    try:
        asyncio.run(dream_mod.run_dream_consolidation(Settings()))
    finally:
        dream_mod.EpisodeQueue = original

    assert captured["path"] == str(_foreign_cwd_and_home / "data" / "queue.db")


def test_cron_does_not_reach_its_db_path_with_contracts_settings(_foreign_cwd_and_home):
    """Locks the measurement that makes cron's change defence-in-depth only.

    If a future ``contracts.PipelineSettings`` grows ``drain_batch_size``,
    this test starts failing and the cron branch becomes genuinely live --
    which is precisely when the home-anchor has to hold.
    """
    import asyncio

    from janus_graph.pipeline.cron import run_cron_sweep

    with pytest.raises(AttributeError, match="drain_batch_size"):
        asyncio.run(run_cron_sweep(Settings()))


def test_home_anchored_db_is_a_real_readable_queue(_foreign_cwd_and_home):
    """The home-anchored DB is a working SQLite queue, not just a path."""
    queue = EpisodeQueue()
    assert sqlite3.connect(str(queue.db_path)).execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'"
    ).fetchone() is not None


# ── FileSink: the site that actually recreated /tmp/data ────────────────


def test_file_sink_default_is_home_anchored(_foreign_cwd_and_home):
    """Bare ``FileSink()`` must not write next to the process CWD.

    Measured 2026-10-02: this was the site still creating ``/tmp/data/logs``
    after the queue defaults were anchored, because
    ``ReportDispatcher.from_settings`` builds a ``FileSink`` from
    ``contracts.ReportSettings.file_path`` (a bare relative Path).
    """
    from janus_graph.report.sinks.file import FileSink

    sink = FileSink()
    assert sink.path == _foreign_cwd_and_home / "data" / "logs" / "janus_report.jsonl"


def test_dispatcher_with_contracts_settings_anchors_file_sink(_foreign_cwd_and_home):
    """End-to-end on the real dispatcher branch that was reachable."""
    from janus_graph.core.contracts import Settings
    from janus_graph.report.dispatcher import ReportDispatcher

    dispatcher = ReportDispatcher.from_settings(Settings())
    file_sinks = [s for s in dispatcher.sinks if type(s).__name__ == "FileSink"]

    assert file_sinks, "expected a FileSink for contracts.Settings"
    assert file_sinks[0].path == _foreign_cwd_and_home / "data" / "logs" / "reports.jsonl"
    assert not (Path(FOREIGN_CWD) / "data").exists()


def test_file_sink_explicit_absolute_path_unchanged(_foreign_cwd_and_home, tmp_path):
    from janus_graph.report.sinks.file import FileSink

    pinned = tmp_path / "pinned" / "r.jsonl"
    assert FileSink(str(pinned)).path == pinned
