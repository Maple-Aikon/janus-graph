"""Repair rule for NodeResolutions schema validation issues."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import HeuristicRule


class NodeResolutionsRule(HeuristicRule):
    """Repairs NodeResolutions payloads.
    
    Handles:
      - {"properties": {"entity_resolutions": [...]}}
      - Bare resolution object {"id": 0, "name": ...} -> wrap in list [{"id": 0, "name": ...}]
      - Bare list of resolutions [{...}] -> {"entity_resolutions": [...]}
      - Key aliases: resolutions, dedupe_results, nodes, entity_resolutions
    """

    @property
    def name(self) -> str:
        return "node_resolutions"

    @property
    def target_model(self) -> str:
        return "NodeResolutions"

    @property
    def target_schema_names(self) -> List[str]:
        return ["NodeResolutions", "EntityResolutions"]

    @property
    def priority(self) -> int:
        return 40

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        return schema_name in self.target_schema_names

    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Dict[str, Any]:
        if isinstance(payload, list):
            return {"entity_resolutions": payload}
        if not isinstance(payload, dict):
            return {"entity_resolutions": []}

        if "properties" in payload and isinstance(payload["properties"], dict):
            payload = {**payload["properties"], **{k: v for k, v in payload.items() if k != "properties"}}

        if "entity_resolutions" in payload:
            return {
                "entity_resolutions": payload["entity_resolutions"]
                if isinstance(payload["entity_resolutions"], list)
                else []
            }

        for src_key in ("resolutions", "dedupe_results", "nodes", "node_resolutions"):
            if src_key in payload:
                val = payload[src_key]
                return {
                    "entity_resolutions": val if isinstance(val, list) else [],
                    **{k: v for k, v in payload.items() if k != src_key},
                }

        # Bare single resolution object
        if {"id", "name"} <= set(payload.keys()):
            return {"entity_resolutions": [payload]}

        return {"entity_resolutions": []}
