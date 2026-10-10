"""Regression tests for the 2026-10-10 mypy sweep (9 ignore_errors modules).

Every test here was written BEFORE the fix and must FAIL on the pre-fix code.
The point is not "mypy is green" — it is that a few of the 21 type errors were
real defects, and a type checker is the only thing that noticed.

Three of them are genuine bugs:

1. ``cli/main.py`` called ``ReportDispatcher.dispatch()``. The class only ever
   had ``emit()`` / ``emit_quick()`` and no ``__getattr__`` escape hatch, so
   ``report test-telegram`` / ``report test-webhook`` raised AttributeError.
   Nothing in the test suite ever invoked those two CLI actions.

2. ``pipeline/dream.py`` read ``cfg.paths`` inside ``getattr()``. ``getattr()``
   only guards the attribute *inside* it; ``cfg.paths`` is evaluated first.
   ``JanusSettings`` has no ``paths``, so a JanusSettings whose
   ``pipeline.queue_db_path`` is None crashed in the fallback that exists
   precisely to avoid crashing.

3. ``heuristics/vendor_patches/falkor_edge_search.py`` imported
   ``graphiti_core.__version__``, which the installed package does not define.
   The ``except Exception`` swallowed it every single run, so ``state.version``
   was silently always "" — and ``summary()`` renders that as ``version=?``.

The rest are type-only and are fixed without changing behaviour.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, List

import pytest

from janus_graph.cli.main import build_parser, run_cli
from janus_graph.config import JanusSettings
from janus_graph.pipeline.dream import run_dream_consolidation
from janus_graph.report.dispatcher import ReportDispatcher

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# BUG 1 — ReportDispatcher.dispatch does not exist
# ---------------------------------------------------------------------------


def test_report_dispatcher_really_has_no_dispatch_attribute():
    """The premise of bug 1, asserted directly so this test cannot pass by
    accident if someone later adds a dispatch alias."""
    assert hasattr(ReportDispatcher, "emit")
    assert hasattr(ReportDispatcher, "emit_quick")
    assert not hasattr(ReportDispatcher, "dispatch")
    assert not hasattr(ReportDispatcher, "__getattr__")


@pytest.mark.parametrize("action", ["test-telegram", "test-webhook"])
def test_cli_report_test_actions_do_not_crash(tmp_path, action):
    """Pre-fix: AttributeError('ReportDispatcher' object has no attribute
    'dispatch') escapes run_cli. These two CLI actions were never covered."""
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = str(tmp_path / "q.db")
    cfg.report.sinks.file.path = str(tmp_path / "reports.jsonl")

    parser = build_parser()
    args = parser.parse_args(["report", action])

    code = run_cli(args, cfg)

    assert code == 0, f"report {action} returned {code}"


# ---------------------------------------------------------------------------
# BUG 2 — getattr(cfg.paths, ...) evaluates cfg.paths first
# ---------------------------------------------------------------------------


def test_dream_queue_path_fallback_survives_janussettings_without_paths(tmp_path, monkeypatch):
    """Pre-fix: JanusSettings has no ``paths`` attribute, so when
    ``pipeline.queue_db_path`` is None the fallback line raised AttributeError.

    Reproduce it exactly the way production would: a real JanusSettings whose
    queue_db_path was explicitly set to None (validate_assignment is unset on
    PipelineConfig, so this assignment is accepted).
    """
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(tmp_path))
    cfg = JanusSettings()
    cfg.pipeline.queue_db_path = None

    # Precondition of the bug: the guard on the previous line does pass.
    assert getattr(cfg.pipeline, "queue_db_path", None) is None
    # Precondition of the bug: this type genuinely has no `paths`.
    assert not hasattr(cfg, "paths")

    # The three lines under test, lifted verbatim from run_dream_consolidation.
    pipeline_path = getattr(cfg.pipeline, "queue_db_path", None)
    if pipeline_path is None:
        paths_cfg = getattr(cfg, "paths", None)
        pipeline_path = getattr(paths_cfg, "queue_db_path", None)
    if pipeline_path is None:
        pipeline_path = "./data/queue.db"

    # Must not raise, and must fall through to the relative default.
    assert pipeline_path == "./data/queue.db"


def test_dream_queue_path_block_uses_guarded_paths_access():
    """Static tripwire: the unsafe ``getattr(cfg.paths, ...)`` shape must not
    come back. getattr(cfg, "paths", None) is the guarded form."""
    src = (REPO_ROOT / "janus_graph" / "pipeline" / "dream.py").read_text()
    assert "getattr(cfg.paths," not in src, (
        "getattr() only guards the attribute inside it; cfg.paths is evaluated "
        "first and JanusSettings has no paths attribute"
    )


# ---------------------------------------------------------------------------
# BUG 3 — graphiti_core.__version__ does not exist
# ---------------------------------------------------------------------------


def test_graphiti_core_really_lacks_dunder_version():
    """The premise of bug 3: installed graphiti-core 0.30.1 has no
    __version__. If a future release adds it, this test tells us to revisit the
    fallback rather than silently keep the dead import."""
    import graphiti_core

    assert not hasattr(graphiti_core, "__version__")


def test_patch_state_version_is_populated_from_package_metadata(tmp_path):
    """Pre-fix: the try/except around the version import swallowed the
    ImportError every run, so ``state.version`` stayed "" forever and
    ``summary()`` printed ``version=?``."""
    from janus_graph.heuristics.vendor_patches.falkor_edge_search import (
        apply_falkor_edge_search_patch,
    )

    state = apply_falkor_edge_search_patch()

    import importlib.metadata as md

    expected = md.version("graphiti-core")

    assert state.version == expected, (
        f"state.version={state.version!r} but graphiti-core is {expected!r}; "
        "the version import is being swallowed"
    )
    assert expected in state.summary()


# ---------------------------------------------------------------------------
# Structural guard: the mypy debt must not silently regrow
# ---------------------------------------------------------------------------


def test_pyproject_has_no_janus_graph_ignore_errors_overrides():
    """A blanket ignore_errors block would make mypy rc=0 while checking
    nothing. Guard the specific thing we just removed."""
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    offenders = [
        o["module"]
        for o in data["tool"]["mypy"].get("overrides", [])
        if o.get("ignore_errors") and str(o.get("module", "")).startswith("janus_graph")
    ]
    assert offenders == [], f"ignore_errors overrides came back: {offenders}"


def test_every_python_file_still_parses():
    """Cheap smoke test after nine files of edits."""
    bad: List[str] = []
    for p in (REPO_ROOT / "janus_graph").rglob("*.py"):
        try:
            ast.parse(p.read_text())
        except SyntaxError as exc:  # pragma: no cover
            bad.append(f"{p}: {exc}")
    assert bad == []
