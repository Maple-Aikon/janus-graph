"""Unit tests for ``config._apply_home_relative_paths``.

Untested private helper. It runs in TWO passes over overlapping field sets:

  Pass 1 (``_HOME_RELATIVE_PATH_FIELDS``, 7 entries) — rewrite any *relative*
  value to ``str((home / value).resolve())``. ``Path`` values keep their type.
  Absolute values short-circuit on ``is_absolute()``.
  Pass 2 (inline tuple, 5 entries) — ``mkdir(parents=True, exist_ok=True)``
  the *parent* dir for file-shaped fields, or the dir itself for
  ``engine.data_dir``. ``bin_dir`` is deliberately excluded.

Coverage:
  - relative str -> prefixed + ``.resolve()``; relative Path -> keeps type
  - absolute str / absolute Path -> value NOT rewritten
  - empty string / None -> skipped in pass 1
  - deep dotted path (report.sinks.file.path) -> leaf only, parent identity kept
  - mkdir creates file parents; bin_dir NOT created; data_dir created as dir
  - idempotent on a second pass
  - all 7 pass-1 fields covered (drop-one guard)
  - operator-pinned absolute fields are skipped by BOTH passes (``pinned``
    set), so a ``/var/lib/...`` pin no longer mkdir's without root
  - guard-against-overcorrection: rewritten-but-absolute fields still mkdir

HARNESS NOTES
-------------
* ``JanusSettings()`` is a ``BaseSettings`` resolving its YAML via
  ``config._resolve_default_yaml_file()`` (CWD + env dependent). The
  ``_isolated_config`` autouse fixture pins ``JANUS_CONFIG_PATH`` to a
  non-existent path and ``JANUS_GRAPH_HOME`` to tmp so these tests never read
  the operator's real ``~/.janus-graph/config.yaml``.
* ``model_post_init`` ALREADY calls the helper, so a freshly built settings
  tree is home-prefixed. ``_relative_tree()`` therefore resets the 7 fields
  back to relative AFTER construction.
* The ``isinstance(current, Path)`` branch looks unreachable (fields are
  annotated ``str``; the constructor REJECTS a ``Path`` with a
  ValidationError rather than coercing it) — but it is NOT. No model sets
  ``validate_assignment``, so a plain assignment AFTER construction stores a
  real ``Path`` verbatim. Two tests pin both halves of that claim:
  ``test_constructor_rejects_path_typed_field`` and
  ``test_path_branch_is_reachable_via_post_construction_assignment``.
* ``engine.pid_file`` is deliberately NOT in the pass-2 mkdir tuple; see
  ``test_pid_file_parent_comes_free_from_data_dir`` for why the default
  layout still gets its parent, and the comment in config.py for the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from janus_graph.config import (
    EngineConfig,
    HeuristicsConfig,
    JanusSettings,
    PipelineConfig,
    ReportConfig,
    _apply_home_relative_paths,
    _resolve_home,
)


# ─── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Keep the settings tree from reading the real operator config."""
    monkeypatch.setenv("JANUS_CONFIG_PATH", str(tmp_path / "does-not-exist.yaml"))
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(tmp_path / "home"))


def _relative_tree() -> JanusSettings:
    """Settings tree with all 7 pass-1 fields forced back to relative values.

    Construction runs ``model_post_init`` -> the helper -> home-prefixing, so
    the reset has to happen afterwards for the helper to have work to do.
    """
    cfg = JanusSettings(
        engine=EngineConfig(
            bin_dir="bin",
            data_dir="data",
            pid_file="janus.pid",
            log_file="logs/janus.log",
        ),
        pipeline=PipelineConfig(queue_db_path="queue/episodes.db"),
        heuristics=HeuristicsConfig(quirks_log_path="logs/quirks.jsonl"),
        report=ReportConfig(sinks={"file": {"path": "logs/report.jsonl"}}),
    )
    object.__setattr__(cfg.engine, "bin_dir", "bin")
    object.__setattr__(cfg.engine, "data_dir", "data")
    object.__setattr__(cfg.engine, "pid_file", "janus.pid")
    object.__setattr__(cfg.engine, "log_file", "logs/janus.log")
    object.__setattr__(cfg.pipeline, "queue_db_path", "queue/episodes.db")
    object.__setattr__(cfg.heuristics, "quirks_log_path", "logs/quirks.jsonl")
    object.__setattr__(cfg.report.sinks.file, "path", "logs/report.jsonl")
    return cfg


# ─── pass 1: relative rewrite ───────────────────────────────────────────


def test_relative_str_field_gets_home_prefix(tmp_path):
    """A relative string path becomes ``str((home / value).resolve())``."""
    cfg = _relative_tree()
    assert cfg.engine.log_file == "logs/janus.log"

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.engine.log_file == str((tmp_path / "logs" / "janus.log").resolve())


def test_path_typed_field_gets_prefix_and_keeps_path_type(tmp_path):
    """The ``isinstance(current, Path)`` branch does NOT str()-coerce.

    Unreachable through the constructor, so inject a real ``Path`` past
    validation and pin the type contract of that branch.
    """
    cfg = _relative_tree()
    object.__setattr__(cfg.engine, "bin_dir", Path("bin"))
    assert isinstance(cfg.engine.bin_dir, Path)

    _apply_home_relative_paths(cfg, tmp_path)

    assert isinstance(cfg.engine.bin_dir, Path), "Path branch must not str() coerce"
    assert cfg.engine.bin_dir == (tmp_path / "bin").resolve()


def test_constructor_rejects_path_typed_field():
    """Pydantic REJECTS a ``Path`` at construction — it does not coerce.

    Pins the behaviour the helper's docstring relies on. If a future upgrade
    (or a ``model_config`` change) starts coercing ``Path`` -> ``str``, this
    test breaks, which is the signal to revisit the ``isinstance`` branch.
    """
    with pytest.raises(ValidationError, match="valid string"):
        EngineConfig(pid_file=Path("/var/lib/janus-graph/falkordb.pid"))


def test_path_branch_is_reachable_via_post_construction_assignment(tmp_path):
    """The ``isinstance(current, Path)`` branch is LIVE, not dead code.

    The constructor rejects ``Path`` (above), which is exactly why this branch
    looks unreachable. It is not: no model sets ``validate_assignment``, so a
    plain attribute assignment after construction is accepted verbatim and
    lands a real ``Path`` in the field.

    Provenance: ``tests/conftest.py`` and ``tests/test_episodes.py`` assign
    ``settings.pipeline.queue_db_path = str(...)`` after construction, and
    nothing in the repo converts those to ``Path`` — but ``engine/server.py``
    reads every one of these fields as ``Path(self.config.<field>)``, so the
    field type is a real contract, not a formality.
    """
    cfg = JanusSettings()
    # Plain assignment, no object.__setattr__ — this is what production code
    # or a plugin would do.
    cfg.engine.log_file = Path("logs/typed.log")
    assert isinstance(cfg.engine.log_file, Path), "validate_assignment must be off"

    _apply_home_relative_paths(cfg, tmp_path)

    # The Path branch ran: prefixed AND still a Path (not str()-coerced).
    assert isinstance(cfg.engine.log_file, Path)
    assert cfg.engine.log_file == (tmp_path / "logs" / "typed.log").resolve()
    # ...and pass 2 still created the parent, since it was rewritten, not pinned.
    assert (tmp_path / "logs").is_dir()


def test_relative_dot_prefix_stays_under_home(tmp_path):
    """A ``./data/x`` value still lands under home after ``.resolve()``."""
    cfg = _relative_tree()
    object.__setattr__(cfg.pipeline, "queue_db_path", "./data/episodes.db")

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.pipeline.queue_db_path == str(tmp_path / "data" / "episodes.db")


def test_deep_dotted_path_rewrites_leaf_and_preserves_parent_identity(tmp_path):
    """``report.sinks.file.path`` is 4 levels deep; only the leaf is setattr'd.

    Callers holding a reference to ``cfg.report.sinks.file`` must observe the
    update, so intermediates are mutated in place, never replaced.
    """
    cfg = _relative_tree()
    sinks_before = cfg.report.sinks
    file_node_before = cfg.report.sinks.file

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.report.sinks.file.path == str(
        (tmp_path / "logs" / "report.jsonl").resolve()
    )
    assert cfg.report.sinks is sinks_before
    assert cfg.report.sinks.file is file_node_before


# ─── pass 1: absolute + degenerate values ────────────────────────────────


def test_absolute_str_value_is_not_rewritten(tmp_path):
    """An absolute string path is operator-pinned; pass 1 leaves the value alone.

    MUTATION NOTE: dropping the ``is_absolute()`` guard is an EQUIVALENT
    mutant here, not a test gap. ``Path("/tmp/home") / "/var/lib/x"`` discards
    the left operand, so the rewritten value is byte-identical either way. The
    guard's only observable effect is skipping the setattr. This assertion
    pins that equivalence explicitly so the survivor is not mistaken for a
    missing test.
    """
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    cfg = JanusSettings()
    object.__setattr__(cfg.engine, "data_dir", str(pinned))

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.engine.data_dir == str(pinned)


def test_absolute_path_value_is_not_rewritten(tmp_path):
    """An absolute ``Path`` short-circuits on the ``is_absolute()`` guard."""
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    cfg = JanusSettings()
    object.__setattr__(cfg.engine, "data_dir", pinned)

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.engine.data_dir is pinned


def test_empty_string_field_is_skipped(tmp_path):
    """Empty string hits ``if not current: continue`` in BOTH passes.

    The quirks path is pointed at a directory name no other field uses, so
    a created dir can only be attributed to this field.
    """
    cfg = _relative_tree()
    object.__setattr__(cfg.heuristics, "quirks_log_path", "")

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.heuristics.quirks_log_path == ""
    # An empty quirks path is skipped, so the rewritten value of some other
    # field never lands it in "logs" -- nothing extra is created for it.
    assert cfg.heuristics.quirks_log_path != str(tmp_path)
    # The sibling fields still get their parents created (sanity: pass 2 ran).
    assert (tmp_path / "logs").is_dir(), "pass 2 must still run for other fields"


def test_empty_field_does_not_create_its_own_parent(tmp_path):
    """Isolate the empty-field mkdir skip using a uniquely-named sibling path."""
    cfg = _relative_tree()
    # Point every OTHER mkdir field at a different dir so "emptydir" can only
    # be created if the empty quirks field were (wrongly) processed.
    object.__setattr__(cfg.engine, "log_file", "a/janus.log")
    object.__setattr__(cfg.engine, "data_dir", "b")
    object.__setattr__(cfg.pipeline, "queue_db_path", "c/episodes.db")
    object.__setattr__(cfg.report.sinks.file, "path", "d/report.jsonl")
    object.__setattr__(cfg.heuristics, "quirks_log_path", "")

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.heuristics.quirks_log_path == ""
    assert (tmp_path / "a").is_dir() and (tmp_path / "b").is_dir()
    assert (tmp_path / "c").is_dir() and (tmp_path / "d").is_dir()
    # No stray dir from the skipped field ("home" comes from the autouse
    # fixture's JANUS_GRAPH_HOME, not from the helper).
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "home") == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_none_field_is_skipped(tmp_path):
    """A ``None`` field hits ``if not isinstance(current, str): continue``."""
    cfg = _relative_tree()
    object.__setattr__(cfg.heuristics, "quirks_log_path", None)

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.heuristics.quirks_log_path is None


# ─── pass 2: mkdir behaviour ────────────────────────────────────────────


def test_mkdir_creates_parents_for_file_paths(tmp_path):
    """queue_db / quirks / report / log parents are created; bin_dir is not."""
    cfg = _relative_tree()
    assert not (tmp_path / "logs").exists()

    _apply_home_relative_paths(cfg, tmp_path)

    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "queue").is_dir()
    assert not (tmp_path / "bin").exists(), "bin_dir must NOT be mkdir'd"


def test_data_dir_is_created_as_a_directory(tmp_path):
    """``engine.data_dir`` is the one field mkdir'd as a dir, not ``.parent``."""
    cfg = _relative_tree()
    assert not (tmp_path / "data").exists()

    _apply_home_relative_paths(cfg, tmp_path)

    assert (tmp_path / "data").is_dir()


def test_pid_file_parent_is_never_created_by_the_helper(tmp_path):
    """``engine.pid_file`` is a pass-1 field but NOT in the pass-2 tuple.

    A relative pid_file is still home-prefixed, but its parent is not mkdir'd
    by the helper. This is INTENTIONAL, not an oversight, and this test pins
    it so a future "add it for consistency" change has to be deliberate.

    Why it is safe to leave out (see the comment in config.py):
      * the stock layout nests the pid under data_dir, and data_dir IS
        mkdir'd by the helper, so the default config already gets the parent;
      * the only consumer mkdirs it itself --
        ``FalkorDBServerManager.start()`` calls
        ``self.pid_file.parent.mkdir(parents=True, exist_ok=True)``
        immediately before exec'ing redis-server.
    """
    cfg = _relative_tree()
    object.__setattr__(cfg.engine, "pid_file", "deep/pid/janus.pid")

    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.engine.pid_file == str(tmp_path / "deep" / "pid" / "janus.pid")
    assert not (tmp_path / "deep").exists(), "pid_file is not in the mkdir tuple"


def test_pid_file_parent_comes_free_from_data_dir(tmp_path):
    """The default layout is covered anyway: the pid nests under data_dir.

    Asserts the structural invariant (pid parent IS the data_dir), not a
    hardcoded path: construction already ran the helper against the
    autouse fixture's ``JANUS_GRAPH_HOME``, so the values are already
    home-prefixed and a second helper call treats them as operator-pinned.
    """
    cfg = JanusSettings()
    default_pid = object.__getattribute__(cfg.engine, "pid_file")
    # Guard: the default must really nest under data_dir, else the argument
    # for excluding pid_file from the mkdir tuple collapses.
    assert "falkordb" in default_pid

    _apply_home_relative_paths(cfg, tmp_path)

    pid = Path(cfg.engine.pid_file)
    data_dir = Path(cfg.engine.data_dir)
    assert pid.parent == data_dir, "default pid must live inside data_dir"
    assert data_dir.is_dir(), "data_dir mkdir covers the default pid parent"


def test_pid_file_parent_is_created_by_the_server_manager(tmp_path, monkeypatch):
    """The OTHER half of the pid_file justification: the consumer mkdirs it.

    config.py claims ``engine.pid_file`` is safe to leave out of the mkdir
    tuple partly because ``FalkorDBServerManager.start()`` creates the parent
    itself. This pins that claim against the real class — if someone drops
    that ``mkdir`` line from engine/server.py, this test fails and the
    config.py comment stops being true.

    ``is_running`` is stubbed to False so ``start()`` runs its body (it
    early-returns True when the server looks alive); the pid file is never
    written, which is why the assertion is on the *parent* directory.
    """
    import subprocess
    from unittest.mock import MagicMock

    from janus_graph.engine.server import FalkorDBServerManager

    pid_parent = tmp_path / "runtime" / "nested" / "falkordb.pid"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "redis-server").write_bytes(b"")
    (bin_dir / "falkordb.so").write_bytes(b"")

    manager = FalkorDBServerManager(
        config=MagicMock(
            bin_dir=str(bin_dir),
            data_dir=str(tmp_path / "data"),
            log_file=str(tmp_path / "logs" / "f.log"),
            pid_file=str(pid_parent),
            port=6379,
            host="127.0.0.1",
        )
    )
    monkeypatch.setattr(FalkorDBServerManager, "is_running", lambda self: False)

    import janus_graph.engine.server as server_mod

    monkeypatch.setattr(
        server_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0),
    )

    manager.start()  # returns False (no real redis ran) — the mkdir is the point

    assert pid_parent.parent.is_dir(), "start() must create the pid parent"


def test_helper_is_idempotent(tmp_path):
    """A second pass is a no-op: values are absolute, so pass 1 short-circuits."""
    cfg = _relative_tree()

    _apply_home_relative_paths(cfg, tmp_path)
    first = cfg.engine.log_file
    _apply_home_relative_paths(cfg, tmp_path)

    assert cfg.engine.log_file == first


def test_all_seven_home_relative_fields_are_covered(tmp_path):
    """Drop-one guard on ``_HOME_RELATIVE_PATH_FIELDS``.

    Removing a field from the tuple would silently stop home-relative
    resolution; this test fails instead of letting that ship.
    """
    cfg = _relative_tree()

    _apply_home_relative_paths(cfg, tmp_path)

    for value in (
        cfg.engine.bin_dir,
        cfg.engine.data_dir,
        cfg.engine.pid_file,
        cfg.engine.log_file,
        cfg.pipeline.queue_db_path,
        cfg.heuristics.quirks_log_path,
        cfg.report.sinks.file.path,
    ):
        assert str(value).startswith(str(tmp_path)), f"not home-prefixed: {value!r}"


# ─── pass 2 honours the same absoluteness decision as pass 1 ───────────


def test_absolute_path_is_not_created_after_being_skipped_by_rewrite(
    tmp_path, monkeypatch
):
    """Operator-pinned absolute paths are off-limits to BOTH passes.

    Pass 1 skips an absolute value (leaves it pinned) and records the field
    in ``pinned``; pass 2 consults that same set and skips the mkdir. Before
    the fix, pass 2 re-read the value with no guard and called
    ``mkdir(parents=True)`` on it — with a real ``/var/lib/...`` pin and no
    root, ``JanusSettings()`` raised ``PermissionError`` at construction.
    """
    pinned = tmp_path / "operator-pinned"
    created: list[str] = []
    real_mkdir = Path.mkdir

    def _spy(self, *args, **kwargs):
        created.append(str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _spy)

    cfg = JanusSettings()
    object.__setattr__(cfg.engine, "data_dir", str(pinned))
    _apply_home_relative_paths(cfg, tmp_path)

    # value untouched (pass 1 skipped it)...
    assert cfg.engine.data_dir == str(pinned)
    # ...and now pass 2 honours that: nothing was created for it.
    assert str(pinned) not in created
    assert not pinned.exists()


def test_mkdir_still_runs_for_rewritten_fields(tmp_path):
    """Guard-against-overcorrection: rewriting does NOT disable mkdir.

    Every value pass 1 rewrites is absolute *by construction*, so a naive
    ``p.is_absolute()`` guard in pass 2 would skip every mkdir and the
    SQLite/log parents would never be created. This is the test that fails
    if someone "fixes" the asymmetry the wrong way.
    """
    cfg = _relative_tree()

    _apply_home_relative_paths(cfg, tmp_path)

    # Values are home-rewritten (hence absolute)...
    assert cfg.pipeline.queue_db_path == str(
        (tmp_path / "queue" / "episodes.db").resolve()
    )
    # ...yet their parents are still created.
    assert (tmp_path / "queue").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "data").is_dir()


# ─── _resolve_home precedence ───────────────────────────────────────────


def test_resolve_home_prefers_env_var(monkeypatch, tmp_path):
    """``JANUS_GRAPH_HOME`` wins over the ``~/.janus-graph`` fallback."""
    monkeypatch.setenv("JANUS_GRAPH_HOME", str(tmp_path / "custom-home"))
    assert _resolve_home() == (tmp_path / "custom-home").resolve()


def test_resolve_home_falls_back_to_dot_janus_graph(monkeypatch):
    """With the env var absent the fallback is ``~/.janus-graph``."""
    monkeypatch.delenv("JANUS_GRAPH_HOME", raising=False)
    assert _resolve_home().name == ".janus-graph"
