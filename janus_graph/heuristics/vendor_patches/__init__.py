"""Runtime patches for pinned vendor code (graphiti_core).

Each patch here is a narrow, anchored, reversible rewrite of a vendored
hot path. They live in janus rather than in site-packages so that
``uv sync`` / ``uv lock --upgrade`` cannot silently drop them.

Current patches:
    falkor_edge_search — remove the :Entity label scan from
    FalkorSearchOperations.edge_fulltext_search. See that module for the
    measured speedup and the upstream issue references.
"""

from .falkor_edge_search import (
    PatchedVendorState,
    apply_falkor_edge_search_patch,
    uninstall_falkor_edge_search_patch,
)

__all__ = [
    "PatchedVendorState",
    "apply_falkor_edge_search_patch",
    "uninstall_falkor_edge_search_patch",
]
