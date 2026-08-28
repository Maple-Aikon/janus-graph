"""Repair rule for ExtractedEdges schema validation issues."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import HeuristicRule


class ExtractedEdgesRule(HeuristicRule):
    """Repairs ExtractedEdges payloads.
    
    Handles:
      - {"properties": {"edges": [...]}} / {"properties": {"facts": [...]}}
      - {"facts": [...]} -> {"edges": [...]}
      - Bare list of edge dicts [{...}] -> {"edges": [...]}
      - Key aliases: facts, fact_triples, edges_list, relations
    """

    @property
    def name(self) -> str:
        return "extracted_edges"

    @property
    def target_model(self) -> str:
        return "ExtractedEdges"

    @property
    def priority(self) -> int:
        return 20

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        return schema_name in self.target_schema_names

    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Dict[str, Any]:
        if isinstance(payload, list):
            return {"edges": payload}
        if not isinstance(payload, dict):
            return {"edges": []}

        if "properties" in payload and isinstance(payload["properties"], dict):
            payload = {**payload["properties"], **{k: v for k, v in payload.items() if k != "properties"}}

        if "edges" in payload:
            return {"edges": payload["edges"] if isinstance(payload["edges"], list) else []}

        for src_key in ("facts", "fact_triples", "edges_list", "relations"):
            if src_key in payload:
                val = payload[src_key]
                return {
                    "edges": val if isinstance(val, list) else [],
                    **{k: v for k, v in payload.items() if k != src_key},
                }

        return {"edges": []}
