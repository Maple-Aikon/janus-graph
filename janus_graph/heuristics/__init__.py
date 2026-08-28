"""Heuristics package for self-repairing LLM outputs and schema resilience."""

from .quirks_logger import QuirksLogger
from .registry import HeuristicRegistry
from .repairing_client import SchemaRepairingLLMClient
from .rules.base import HeuristicRule
from .rules.edge_duplicate import EdgeDuplicateRule
from .rules.extracted_edges import ExtractedEdgesRule
from .rules.node_resolutions import NodeResolutionsRule

__all__ = [
    "SchemaRepairingLLMClient",
    "QuirksLogger",
    "HeuristicRegistry",
    "HeuristicRule",
    "EdgeDuplicateRule",
    "ExtractedEdgesRule",
    "NodeResolutionsRule",
]
