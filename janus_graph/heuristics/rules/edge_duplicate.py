"""Repair rule for EdgeDuplicate schema validation issues."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import HeuristicRule


def _sanitize_int_list(val: Any) -> List[int]:
    """Coerce any LLM output (None, string, int, list with strings/None) into a clean list[int]."""
    if val is None:
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, (list, tuple, set)):
        res: List[int] = []
        for x in val:
            if x is None:
                continue
            try:
                res.append(int(x))
            except (ValueError, TypeError):
                continue
        return res
    if isinstance(val, (float,)):
        return [int(val)]
    if isinstance(val, str):
        clean = val.strip("[](){}\"' ")
        if not clean or clean.lower() in ("none", "null", "empty", "[]", "n/a", "no", "false"):
            return []
        res = []
        for part in clean.replace(";", ",").split(","):
            part_clean = part.strip()
            if part_clean:
                try:
                    res.append(int(part_clean))
                except (ValueError, TypeError):
                    pass
        return res
    return []


class EdgeDuplicateRule(HeuristicRule):
    """Repairs EdgeDuplicate payloads where list fields are missing or mis-typed.
    
    Handles:
      - {"properties": {"duplicate_facts": [...], "contradicted_facts": [...]}}
      - Missing one or both of duplicate_facts/contradicted_facts
      - Natural language aliases (duplicate_indices, contradicted_indices, etc.)
      - Raw strings / mixed integers
    """

    @property
    def name(self) -> str:
        return "edge_duplicate"

    @property
    def target_model(self) -> str:
        return "EdgeDuplicate"

    @property
    def priority(self) -> int:
        return 10

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        return schema_name in self.target_schema_names

    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"duplicate_facts": [], "contradicted_facts": []}

        # Unwrap nested 'properties' wrapper if present
        if "properties" in payload and isinstance(payload["properties"], dict):
            payload = {**payload["properties"], **{k: v for k, v in payload.items() if k != "properties"}}

        out: Dict[str, Any] = {}

        # duplicate_facts: try multiple key aliases
        dup_val = None
        if "duplicate_facts" in payload:
            dup_val = payload["duplicate_facts"]
        else:
            for src_key in ("duplicate_indices", "duplicates", "dup_idx", "duplicate_fact_ids", "duplicate_ids"):
                if src_key in payload:
                    dup_val = payload[src_key]
                    break

        out["duplicate_facts"] = _sanitize_int_list(dup_val)

        # contradicted_facts: try multiple key aliases
        contra_val = None
        if "contradicted_facts" in payload:
            contra_val = payload["contradicted_facts"]
        else:
            for src_key in ("contradicted_indices", "contradictions", "contra_idx", "contradicted_fact_ids", "contradicted_ids"):
                if src_key in payload:
                    contra_val = payload[src_key]
                    break

        out["contradicted_facts"] = _sanitize_int_list(contra_val)

        return out
