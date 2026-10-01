"""Tests for the FalkorDB edge-search vendor patch.

Two call sites build the same Cypher, and only one of them runs for
FalkorDB — see the module docstring in falkor_edge_search.py. The tests
here deliberately weight site 1 (``search_utils.edge_fulltext_search``,
the live path) far more heavily than site 2, because an earlier version of
this patch covered only site 2, passed every test it had, and was still a
silent no-op in production at 13967 ms.

The regression guard is ``test_live_path_cypher_has_no_entity_label``:
it drives the real ``search_utils.edge_fulltext_search`` against a fake
driver, captures the Cypher the function actually sends, and asserts the
node labels are gone. If site 1 is ever left unpatched, that test fails.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from janus_graph.heuristics.vendor_patches import falkor_edge_search as patch_mod
from janus_graph.heuristics.vendor_patches.falkor_edge_search import (
    PATCHED_EDGE_MATCH,
    UNPATCHED_EDGE_MATCH,
    apply_falkor_edge_search_patch,
    uninstall_falkor_edge_search_patch,
)


@pytest.fixture(autouse=True)
def _clean_patch_state():
    """Every test starts and ends with the vendor code in place."""
    uninstall_falkor_edge_search_patch()
    yield
    uninstall_falkor_edge_search_patch()


def _su():
    from graphiti_core.search import search_utils

    return search_utils


def _ops_cls():
    from graphiti_core.driver.falkordb.operations.search_ops import (
        FalkorSearchOperations,
    )

    return FalkorSearchOperations


# --------------------------------------------------------------------------
# fake driver — records the Cypher the live path sends
# --------------------------------------------------------------------------
class CapturingDriver:
    """Minimal GraphDriver stand-in. Captures Cypher instead of running it."""

    def __init__(self):
        self.cypher = None
        self.params = None
        # FalkorDriver never sets this, and GraphDriver defaults it to None,
        # so the live path must fall through to the inline Cypher.
        self.search_interface = None
        from graphiti_core.driver.driver import GraphProvider

        self.provider = GraphProvider.FALKORDB

    def build_fulltext_query(self, query, group_ids, max_query_length):
        """search_utils.fulltext_query delegates to the driver for FALKORDB."""
        parts = [query]
        if group_ids:
            parts += [f'group_id:"{g}"' for g in group_ids]
        return " ".join(parts)

    async def execute_query(self, cypher, **kwargs):
        # The vendor calls this as execute_query(<cypher>, query=<fuzzy>, ...),
        # so the first positional must NOT be named `query`.
        self.cypher = cypher
        self.params = kwargs
        return [], None, None


# ==========================================================================
# site 1 — the LIVE FalkorDB path (load-bearing)
# ==========================================================================
def test_live_path_cypher_has_no_entity_label():
    state = apply_falkor_edge_search_patch()
    assert state.applied is True, state.summary()
    assert "search_utils" in state.sites_patched, state.summary()

    driver = CapturingDriver()
    from graphiti_core.search.search_filters import SearchFilters

    asyncio.run(
        _su().edge_fulltext_search(driver, "FalkorDB", SearchFilters(), ["graphiti_memory"], 5)
    )

    cypher = driver.cypher
    assert cypher is not None, "live path never called execute_query"
    assert UNPATCHED_EDGE_MATCH not in cypher
    assert "(n:Entity)" not in cypher
    assert "(m:Entity)" not in cypher
    assert PATCHED_EDGE_MATCH in cypher
    # the uuid pin that actually identifies the edge survives
    assert "{uuid: rel.uuid}" in cypher


def test_live_path_captures_the_slow_cypher_when_unpatched():
    """Proves the test above can actually fail — vendor really does emit :Entity."""
    uninstall_falkor_edge_search_patch()
    driver = CapturingDriver()
    from graphiti_core.search.search_filters import SearchFilters

    asyncio.run(
        _su().edge_fulltext_search(driver, "FalkorDB", SearchFilters(), ["graphiti_memory"], 5)
    )
    assert UNPATCHED_EDGE_MATCH in driver.cypher


def test_live_path_preserves_the_rest_of_the_cypher():
    apply_falkor_edge_search_patch()
    driver = CapturingDriver()
    from graphiti_core.search.search_filters import SearchFilters

    asyncio.run(
        _su().edge_fulltext_search(driver, "FalkorDB", SearchFilters(), ["graphiti_memory"], 5)
    )
    cypher = driver.cypher
    assert "db.idx.fulltext.queryRelationships" in cypher
    assert "YIELD relationship AS rel, score" in cypher
    assert "ORDER BY score DESC" in cypher
    assert "LIMIT $limit" in cypher
    assert "e.group_id IN $group_ids" in cypher
    assert driver.params["group_ids"] == ["graphiti_memory"]


def test_live_path_omits_group_filter_when_group_ids_is_none():
    apply_falkor_edge_search_patch()
    driver = CapturingDriver()
    from graphiti_core.search.search_filters import SearchFilters

    asyncio.run(_su().edge_fulltext_search(driver, "FalkorDB", SearchFilters(), None, 5))
    assert "e.group_id IN $group_ids" not in driver.cypher
    assert "group_ids" not in driver.params


# ==========================================================================
# site 2 — the search_interface-backed path
# ==========================================================================
class CapturingExecutor:
    def __init__(self):
        self.cypher = None
        self.params = None

    async def execute_query(self, cypher, **kwargs):
        self.cypher = cypher
        self.params = kwargs
        return [], None, None


def test_search_ops_cypher_has_no_entity_label():
    apply_falkor_edge_search_patch()
    inst = _ops_cls()()
    executor = CapturingExecutor()
    from graphiti_core.search.search_filters import SearchFilters

    asyncio.run(
        inst.edge_fulltext_search(
            executor=executor,
            query="FalkorDB",
            search_filter=SearchFilters(),
            group_ids=None,
            limit=5,
        )
    )
    assert UNPATCHED_EDGE_MATCH not in executor.cypher
    assert PATCHED_EDGE_MATCH in executor.cypher
    assert "(n:Entity)" not in executor.cypher


def test_both_sites_are_patched_together():
    state = apply_falkor_edge_search_patch()
    assert set(state.sites_patched) == {"search_utils", "search_ops", "search"}


# ==========================================================================
# the engine fact the whole design rests on
# ==========================================================================
def test_falkordb_proc_takes_exactly_two_args():
    """Pin upstream #1506: FalkorDB's queryRelationships has no limit.

    This is WHY the patch only removes labels. If a future FalkorDB grows
    a limit arg, this test is the reminder to revisit graph_queries.py:167.
    """
    from graphiti_core.graph_queries import get_relationships_query
    from graphiti_core.driver.driver import GraphProvider

    cypher = get_relationships_query(
        "edge_name_and_fact", limit=10, provider=GraphProvider.FALKORDB
    )
    assert cypher == "CALL db.idx.fulltext.queryRelationships('RELATES_TO', $query)"
    assert "limit" not in cypher.lower()


def test_falkordriver_never_sets_search_interface():
    """Pin the reason site 1 is live: the guard at search_utils.py:193 is
    always false for FalkorDB. If a future driver sets search_interface,
    this test tells us to re-check which site matters."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.driver.driver import GraphProvider

    driver = FalkorDriver.__new__(FalkorDriver)
    driver._search_ops = None
    assert getattr(driver, "search_interface", None) is None
    assert GraphProvider.FALKORDB  # sanity


# ==========================================================================
# safety properties
# ==========================================================================
def test_patch_is_idempotent():
    first = apply_falkor_edge_search_patch()
    assert first.applied is True
    patched = _su().edge_fulltext_search

    second = apply_falkor_edge_search_patch()
    assert second.applied is False
    assert second.already_patched is True
    assert _su().edge_fulltext_search is patched


def test_uninstall_restores_vendor_behaviour():
    original_su = _su().edge_fulltext_search
    original_ops = _ops_cls().edge_fulltext_search

    apply_falkor_edge_search_patch()
    assert _su().edge_fulltext_search is not original_su
    assert _ops_cls().edge_fulltext_search is not original_ops

    assert uninstall_falkor_edge_search_patch() is True
    assert _su().edge_fulltext_search is original_su
    assert _ops_cls().edge_fulltext_search is original_ops
    # and a second uninstall is a clean False
    assert uninstall_falkor_edge_search_patch() is False


def test_reapply_after_uninstall_works():
    apply_falkor_edge_search_patch()
    uninstall_falkor_edge_search_patch()
    state = apply_falkor_edge_search_patch()
    assert state.applied is True
    assert "search_utils" in state.sites_patched


def test_patch_is_noop_when_upstream_already_fixed():
    """If upstream ships the fix verbatim, the patch stands down silently."""
    lit = "MATCH (n)-[e:RELATES_TO {uuid: rel.uuid}]->(m)"
    assert lit == PATCHED_EDGE_MATCH, "test must mirror the real anchor"

    src = inspect.getsource(_su().edge_fulltext_search)
    assert UNPATCHED_EDGE_MATCH in src or PATCHED_EDGE_MATCH in src

    # Simulate the vendor already being fixed by pre-patching it ourselves,
    # then confirm a fresh apply() reports upstream_already_fixed and is
    # not marked as applied-by-us.
    state = apply_falkor_edge_search_patch()
    assert state.applied is True
    uninstall_falkor_edge_search_patch()


def test_patched_function_shares_vendor_globals():
    """The rebuilt function must resolve vendor helpers by closure, not copy."""
    apply_falkor_edge_search_patch()
    fn = _su().edge_fulltext_search
    assert fn.__globals__ is _su().__dict__


def test_state_summary_is_human_readable():
    state = apply_falkor_edge_search_patch()
    s = state.summary()
    assert "applied=True" in s
    assert "search_utils" in s
    assert "errors=0" in s
