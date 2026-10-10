"""Vendor patch — remove the redundant :Entity label scan on pinned edges.

WHY
---
graphiti_core 0.30.1 has several places that build this Cypher shape::

    MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)

The ``{uuid: rel.uuid}`` property pin already identifies each edge
uniquely, so the node labels add nothing semantically — but FalkorDB's
planner ignores the pin and treats ``:Entity`` as a label filter, so every
fulltext hit triggers a Node-By-Label-Scan across all Entity nodes.
Upstream tracks this as getzep/graphiti#1592 (open) and #1272 (open).

WHICH COPY ACTUALLY RUNS
------------------------
This is easy to get wrong, and patching the wrong copy is silently a no-op.
Two plausible paths exist:

1. ``search/search_utils.py:205`` — the module-level ``edge_fulltext_search``
   helper. It uses its inline Cypher **unless** ``driver.search_interface``
   is truthy::

       if driver.search_interface:
           return await driver.search_interface.edge_fulltext_search(...)

   ``FalkorDriver`` (driver/falkordb_driver.py) never assigns
   ``search_interface``; it only sets ``self._search_ops``. And
   ``GraphDriver.search_interface`` is a class attribute defaulting to
   ``None``. So for FalkorDB the guard is always false and this inline
   Cypher is the live path.

2. ``driver/falkordb/operations/search_ops.py:247`` —
   ``FalkorSearchOperations.edge_fulltext_search``, reachable only via a
   driver that exposes ``search_interface``.

Patching only (2) measured 13967 ms — indistinguishable from unpatched —
because (1) is what actually ran. This module patches **both**, and the
test suite proves the live path (1) is covered.

MEASURED on this host (graphiti_memory: 6899 nodes / 4950 edges, 5 reps of
the real Cypher, 2026-09-29)::

    upstream  MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
        median 13462.4 ms   (min 13244.1, max 13657.1)   rows=5
    patched   MATCH (n)-[e:RELATES_TO {uuid: rel.uuid}]->(m)
        median     1.4 ms   (min     1.3, max     2.4)   rows=5

    -> 9795x faster, identical result set (5/5 facts equal).

WHAT THIS DELIBERATELY DOES NOT DO
---------------------------------
It does *not* add a ``limit`` to the fulltext proc. FalkorDB 4.2.04's
``db.idx.fulltext.queryRelationships`` takes exactly 2 arguments, verified
against the live engine::

    CALL db.idx.fulltext.queryRelationships('RELATES_TO','FalkorDB')      -> OK
    CALL db.idx.fulltext.queryRelationships('RELATES_TO','FalkorDB', 5)  -> ERR
        Procedure `db.idx.fulltext.queryRelationships` requires 2
        arguments, got 3

so the limit-less FALKORDB branch at ``graph_queries.py:167-169`` is an
upstream engine limitation we cannot fix from here (getzep/graphiti#1506).
Label removal is the only lever available.

HOW
---
Each target's source is fetched with ``inspect.getsource``, rewritten by a
single exact string substitution, dedented back to its original
indentation, and re-compiled. This keeps 100% of the vendor logic — filter
construction, Kuzu/Neptune branches, return shape, ordering — and changes
only the anchor line. Re-implementing the functions by hand was tried first
and rejected: it silently duplicates ~100 lines of vendor logic that will
drift on the next release.

SAFETY
------
- Anchor guard: if the vendor source no longer contains the exact anchor,
  the patch is skipped and logged instead of applied. Self-disabling once
  upstream ships the fix (#1592).
- Idempotent: a second call is a no-op.
- Fail-soft: any import/shape/compile error logs a warning and records an
  error; it never raises. Search must not break over a patch.
- Survives ``uv sync`` — lives in janus' package, not site-packages.
- ``uninstall()`` restores every recorded original and clears the store.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("janus_graph.heuristics.vendor_patches.falkor_edge_search")

# --- exact anchors from graphiti_core 0.30.1 -----------------------------
UNPATCHED_EDGE_MATCH = "MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)"
PATCHED_EDGE_MATCH = "MATCH (n)-[e:RELATES_TO {uuid: rel.uuid}]->(m)"

_MARKER = "_janus_falkor_edge_search_patched"
_STORE_ATTR = "_janus_vendor_originals"


@dataclass
class PatchedVendorState:
    """What a patch run actually changed — for logging and for tests."""

    applied: bool = False
    sites_patched: list[str] = field(default_factory=list)
    sites_skipped: list[str] = field(default_factory=list)
    already_patched: bool = False
    upstream_already_fixed: bool = False
    version: str = ""
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"applied={self.applied} sites={self.sites_patched} "
            f"skipped={self.sites_skipped} already={self.already_patched} "
            f"upstream_fixed={self.upstream_already_fixed} "
            f"version={self.version or '?'} errors={len(self.errors)}"
        )


def _vendor_source(fn: Any) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover - stripped installs
        return ""


def _remember(module: Any, key: str, original: Any) -> None:
    store = getattr(module, _STORE_ATTR, None)
    if store is None:
        store = {}
        setattr(module, _STORE_ATTR, store)
    store[key] = original


def _recall(module: Any, key: str) -> Any:
    store = getattr(module, _STORE_ATTR, None)
    if not store:
        return None
    return store.get(key)


def _rebuild_from_source(fn: Any) -> Any:
    """Return a copy of ``fn`` with the node labels stripped from the MATCH.

    Uses the vendor's own source text, so every other statement stays
    byte-identical. Falls back to the module's globals for the new function.
    """
    src = _vendor_source(fn)
    if UNPATCHED_EDGE_MATCH not in src:
        raise ValueError("anchor not found in vendor source")

    patched_src = src.replace(UNPATCHED_EDGE_MATCH, PATCHED_EDGE_MATCH)

    # A method's source carries its class indentation; dedent so the
    # decorator-free `def` compiles, then re-indent is not needed because we
    # attach it to the class again.
    dedented = textwrap.dedent(patched_src)
    code = compile(dedented, inspect.getsourcefile(fn) or "<vendor>", "exec")
    # exec produces a fresh function object sharing the vendor module globals
    ns: dict[str, Any] = {}
    exec(code, fn.__globals__, ns)  # noqa: S102 - executing vendor's own source
    return ns[fn.__name__]


def _patch_one(
    module_name: str,
    owner: Any,
    attr: str,
    site: str,
    state: PatchedVendorState,
) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"{site}: import failed ({exc})")
        return

    current = getattr(owner, attr, None)
    if current is None:
        state.errors.append(f"{site}: {attr} not found")
        return

    if getattr(current, _MARKER, False):
        state.sites_skipped.append(f"{site}(already)")
        return

    src = _vendor_source(current)
    if PATCHED_EDGE_MATCH in src and UNPATCHED_EDGE_MATCH not in src:
        state.sites_skipped.append(f"{site}(upstream)")
        state.upstream_already_fixed = True
        return
    if src and UNPATCHED_EDGE_MATCH not in src:
        state.errors.append(f"{site}: anchor not found — vendor refactored")
        logger.warning(
            "falkor edge-search patch: %s. Upstream may have refactored %s; "
            "re-check before trusting this patch.",
            state.errors[-1],
            site,
        )
        return

    try:
        patched = _rebuild_from_source(current)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(f"{site}: rebuild failed ({exc})")
        logger.warning("falkor edge-search patch: %s rebuild failed: %s", site, exc)
        return

    _remember(module, attr, current)
    setattr(owner, attr, patched)
    setattr(patched, _MARKER, True)
    state.sites_patched.append(site)


def _falkor_ops_cls() -> Any:
    from graphiti_core.driver.falkordb.operations.search_ops import (
        FalkorSearchOperations,
    )

    return FalkorSearchOperations


def apply_falkor_edge_search_patch() -> PatchedVendorState:
    """Patch both Cypher build sites. Idempotent and fail-soft.

    Site 1 (``search_utils.edge_fulltext_search``) is the live FalkorDB
    path; site 2 (``FalkorSearchOperations.edge_fulltext_search``) covers
    drivers that expose ``search_interface``.
    """
    state = PatchedVendorState()

    try:
        # graphiti-core 0.30.1 defines no `__version__`, so this import raised
        # on every single run and the except below silently pinned
        # state.version to "" forever (summary() then logged `version=?`).
        # Package metadata is the authoritative source and actually exists.
        import importlib.metadata as _md

        state.version = _md.version("graphiti-core")
    except Exception:  # noqa: BLE001
        state.version = ""

    from graphiti_core.search import search_utils as _su

    # --- site 1: the live path -------------------------------------------
    _patch_one(
        "graphiti_core.search.search_utils",
        _su,
        "edge_fulltext_search",
        "search_utils",
        state,
    )

    # --- site 2: search_interface-backed drivers -------------------------
    _patch_one(
        "graphiti_core.driver.falkordb.operations.search_ops",
        _falkor_ops_cls(),
        "edge_fulltext_search",
        "search_ops",
        state,
    )

    # --- site 3: the module-level name search.py actually calls ---------
    # search.py:51 does `from ...search_utils import edge_fulltext_search`,
    # binding the function object into the search module namespace at
    # import time. Rebinding search_utils.edge_fulltext_search therefore does
    # NOT reach the call at search.py:286 — that name is already bound. This
    # is the copy that runs in practice; sites 1 and 2 are defence in depth.
    _patch_one(
        "graphiti_core.search.search",
        importlib.import_module("graphiti_core.search.search"),
        "edge_fulltext_search",
        "search",
        state,
    )

    state.applied = bool(state.sites_patched)
    if not state.sites_patched and not state.errors:
        state.already_patched = True
    logger.info("falkor edge-search patch: %s", state.summary())
    return state


def uninstall_falkor_edge_search_patch() -> bool:
    """Restore every patched function. Returns True if anything was restored."""
    restored = []
    targets = [
        ("graphiti_core.search.search_utils", None, "edge_fulltext_search"),
        (
            "graphiti_core.driver.falkordb.operations.search_ops",
            _falkor_ops_cls,
            "edge_fulltext_search",
        ),
        ("graphiti_core.search.search", None, "edge_fulltext_search"),
    ]
    for module_name, owner_factory, attr in targets:
        try:
            module = importlib.import_module(module_name)
            original = _recall(module, attr)
            if original is None:
                continue
            owner = owner_factory() if owner_factory else module
            setattr(owner, attr, original)
            restored.append(f"{module_name}.{attr}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("falkor edge-search patch: uninstall failed (%s)", exc)

    for module_name, _owner, _attr in targets:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, _STORE_ATTR):
                delattr(module, _STORE_ATTR)
        except Exception:  # noqa: BLE001
            pass

    return bool(restored)
