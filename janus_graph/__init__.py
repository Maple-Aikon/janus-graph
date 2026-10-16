"""Janus-Graph: Modular, Resilient Knowledge Graph Memory Engine.

Version Single Source of Truth (SoT)
------------------------------------
``pyproject.toml`` -> ``[project] version`` is the ONLY place a janus-graph
version number is written. Every report site (HTTP ``/health``, MCP
``graphiti_health``, ``janus_graph.__version__``) derives from it.

Why not ``importlib.metadata``:
    The install is **editable**, so ``site-packages/janus_graph-<v>.dist-info/
    METADATA`` is a *build-time snapshot*. It still said ``0.1.0`` long after
    pyproject moved on, which is exactly how three different numbers ended up
    being reported by three different endpoints. Reading the TOML directly
    makes the stale metadata copy irrelevant instead of authoritative.

Why not hardcode: that is the bug this module exists to prevent (three literals:
``__init__`` 0.1.0, ``daemon/http_server`` 0.5.0, ``mcp/server`` 0.1.0).

Bumping the version = edit one line in pyproject.toml. Do not add a second
literal anywhere.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__author__ = "Maple Aikon"

#: Emitted when the version cannot be resolved from pyproject.toml. Kept as an
#: obviously-fake sentinel (NOT a plausible version) so a broken install shows
#: up in /health instead of silently reporting a stale-but-valid number.
VERSION_UNRESOLVED = "0.0.0+unresolved"


def _find_pyproject() -> Path | None:
    """Walk up from this file looking for janus-graph's own pyproject.toml.

    Editable installs point at the source tree, so walking up from ``__file__``
    finds the real file. The ``name`` guard stops us from latching onto a
    parent directory's pyproject (a monorepo root, a workspace root).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        project = data.get("project") or {}
        if project.get("name") == "janus-graph" and project.get("version"):
            return candidate
    return None


def _resolve_version() -> str:
    pyproject = _find_pyproject()
    if pyproject is None:
        return VERSION_UNRESOLVED
    try:
        with pyproject.open("rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return VERSION_UNRESOLVED


#: The one derived version. Never assign a literal to this name.
__version__ = _resolve_version()

#: Absolute path to the SoT file, so a bug report can name it precisely.
__version_source__ = str(_find_pyproject()) if _find_pyproject() else "<unresolved>"

__all__ = ["__author__", "__version__", "__version_source__", "VERSION_UNRESOLVED"]
