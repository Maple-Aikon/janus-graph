"""Repair rule for ExtractedEntities schema validation issues."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import HeuristicRule


class ExtractedEntitiesRule(HeuristicRule):
    """Repairs ExtractedEntities payloads.
    
    Handles:
      - {"properties": {"extracted_entities": [...]}}
      - {"entities": [...]} -> {"extracted_entities": [...]}
      - Bare list of entity dicts [{...}] -> {"extracted_entities": [...]}
      - Key aliases: entities, node_list, extracted, items
    """

    @property
    def name(self) -> str:
        return "extracted_entities"

    @property
    def target_model(self) -> str:
        return "ExtractedEntities"

    @property
    def priority(self) -> int:
        return 30

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        return schema_name in self.target_schema_names

    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Dict[str, Any]:
        if isinstance(payload, list):
            return {"extracted_entities": payload}
        if not isinstance(payload, dict):
            return {"extracted_entities": []}

        if "properties" in payload and isinstance(payload["properties"], dict):
            payload = {**payload["properties"], **{k: v for k, v in payload.items() if k != "properties"}}

        if "extracted_entities" in payload:
            return {
                "extracted_entities": payload["extracted_entities"]
                if isinstance(payload["extracted_entities"], list)
                else []
            }

        for src_key in ("entities", "node_list", "extracted", "items"):
            if src_key in payload:
                val = payload[src_key]
                return {
                    "extracted_entities": val if isinstance(val, list) else [],
                    **{k: v for k, v in payload.items() if k != src_key},
                }

        return {"extracted_entities": []}
