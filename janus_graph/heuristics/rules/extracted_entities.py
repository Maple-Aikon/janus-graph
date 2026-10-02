"""Repair rule for ExtractedEntities and SummarizedEntities schema validation issues."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import HeuristicRule

# Wrapper keys the LLM uses when it returns a *container* instead of the
# validated payload. "items" is included because the DLQ shows a JSON-Schema
# echo shape {"items": [...], "type": "array"} (2026-09-30).
_EXTRACTED_LIST_KEYS: Tuple[str, ...] = (
    "extracted_entities",
    "entities",
    "node_list",
    "extracted",
    "items",
)
_SUMMARY_LIST_KEYS: Tuple[str, ...] = (
    "summaries",
    "summarized_entities",
    "summary_list",
    "items",
)

# The LLM sometimes echoes the Pydantic *class name* as the wrapper key, e.g.
# {"SummarizedEntity": [...]}. The value is the real list; the key is not a
# field name and must not survive into the repaired payload.
_CLASS_NAME_KEYS: Tuple[str, ...] = (
    "SummarizedEntity",
    "ExtractedEntity",
    "SummarizedEntities",
    "ExtractedEntities",
)

# Field names that mark a bare dict as a single entity rather than a wrapper.
_SINGLE_ENTITY_MARKERS: Tuple[str, ...] = ("name", "summary", "entity_type", "entity_type_id")


def _coerce_indices(value: Any) -> List[int]:
    """Coerce whatever the LLM gave for episode_indices into list[int]."""
    if isinstance(value, list):
        out: List[int] = []
        for v in value:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out or [0]
    if isinstance(value, bool):
        return [0]
    if isinstance(value, (int, float)):
        return [int(value)]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return [0]


def _normalize_extracted_entity(entity: Any) -> Dict[str, Any]:
    """Coerce one LLM entity dict into ExtractedEntity shape.

    ExtractedEntity (graphiti_core.prompts.extract_nodes) requires:
      - name: str (required)
      - entity_type_id: int (required)
      - episode_indices: list[int] (defaults to [0])

    Drift observed in the DLQ on 2026-09-15: the LLM returns a string
    ``entity_type`` label instead of the integer ``entity_type_id``. Defaulting
    to 0 keeps the episode alive; the string label is preserved so a later
    re-classification pass can still use it.
    """
    if not isinstance(entity, dict):
        return {"name": str(entity), "entity_type_id": 0, "episode_indices": [0]}

    out = dict(entity)
    raw_type_id = out.get("entity_type_id")
    if "entity_type_id" not in out or isinstance(raw_type_id, bool):
        # bool is a subclass of int, so int(True) would silently classify the
        # entity as type 1. A boolean is never a real entity_type_id.
        out["entity_type_id"] = 0
    elif not isinstance(raw_type_id, int):
        try:
            out["entity_type_id"] = int(raw_type_id)
        except (TypeError, ValueError):
            out["entity_type_id"] = 0
    out["episode_indices"] = _coerce_indices(out.get("episode_indices", [0]))
    if not out.get("name"):
        out["name"] = "<unnamed>"
    return out


def _normalize_summarized_entity(entity: Any) -> Dict[str, Any]:
    """Coerce one LLM summary dict into SummarizedEntity shape.

    SummarizedEntity (graphiti_core.prompts.extract_nodes) requires:
      - name: str (required)
      - summary: str (required)
    """
    if not isinstance(entity, dict):
        # A bare string is the summary text for an entity the LLM left unnamed.
        return {"name": "<unnamed>", "summary": "" if entity is None else str(entity)}
    out = dict(entity)
    if not out.get("name"):
        out["name"] = "<unnamed>"
    if not out.get("summary"):
        out["summary"] = ""
    return out


class ExtractedEntitiesRule(HeuristicRule):
    """Repairs ExtractedEntities AND SummarizedEntities payloads.

    Handles, for both schemas:
      - {"properties": {...}} wrapper unwrap
      - key aliases: extracted_entities/entities/node_list/extracted/items
        and summaries/summarized_entities/summary_list/items
      - class-name wrappers, e.g. {"SummarizedEntity": [...]}
      - a bare list of entity dicts
      - a single entity dict — pydantic reports the *inner* value in
        ``err.errors()[0]["input"]``, so the caller hands us an entity rather
        than the envelope
      - a raw JSON string

    Why this rule exists (2026-09-15, reworked 2026-10-02): schema drift was
    the second-largest DLQ class. SummarizedEntities had NO rule at all, so
    ``HeuristicRegistry.find_rule`` returned None and the caller re-raised the
    original ValidationError every time — 21 rows sat in the DLQ unrepairable.

    Empty-list guarantee: a bare ``{}`` (or a JSON-Schema ``$defs`` echo) is
    the one case that legitimately repairs to an empty list — there is no
    entity data in it to lose. Any payload that *does* carry entities must
    produce a non-empty list, otherwise the repair would pass
    ``model_validate`` while silently discarding them.
    """

    @property
    def name(self) -> str:
        return "extracted_entities"

    @property
    def target_model(self) -> str:
        return "ExtractedEntities"

    @property
    def target_schema_names(self) -> List[str]:
        # One rule services both schemas — they share the same drift shapes
        # and differ only in the output key and the required inner fields.
        return ["ExtractedEntities", "SummarizedEntities"]

    @property
    def priority(self) -> int:
        return 30

    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        return schema_name in self.target_schema_names

    @staticmethod
    def _find_container(payload: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[List[Any]]:
        """Return the first list-shaped value under a known key, or None."""
        for src_key in keys:
            if src_key not in payload:
                continue
            val = payload[src_key]
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                # A single entity parked under a list key.
                return [val]
            if isinstance(val, (set, frozenset, tuple)):
                # Defensive: the DLQ of 2026-09-30 holds rows whose value is a
                # one-element set rather than a list or dict. The origin of
                # that set is not understood, but the data is recoverable, so
                # treat it as a container instead of dropping it.
                return list(val)
        return None

    def repair(
        self, schema_name: str, payload: Any, error: Optional[Exception] = None
    ) -> Dict[str, Any]:
        is_summaries = schema_name == "SummarizedEntities"
        default_key = "summaries" if is_summaries else "extracted_entities"
        normalize = _normalize_summarized_entity if is_summaries else _normalize_extracted_entity

        def done(items: List[Any]) -> Dict[str, Any]:
            return {default_key: [normalize(e) for e in items]}

        # 1. Raw JSON string — the LLM returned text instead of an object.
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped.startswith(("{", "[")):
                return {default_key: []}
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                return {default_key: []}

        # 2. Bare list of entity dicts.
        if isinstance(payload, list):
            return done(payload)

        # 3. Anything that is not a mapping is unrecoverable.
        if not isinstance(payload, dict):
            return {default_key: []}

        # 4. Unwrap {"properties": {...}}.
        if isinstance(payload.get("properties"), dict):
            payload = {
                **payload["properties"],
                **{k: v for k, v in payload.items() if k != "properties"},
            }

        # 5. Known container keys.
        keys = _SUMMARY_LIST_KEYS if is_summaries else _EXTRACTED_LIST_KEYS
        items = self._find_container(payload, keys)
        if items is not None:
            return done(items)

        # 6. Class-name wrapper, e.g. {"SummarizedEntity": [...]}. The value is
        #    sometimes a bare string — the LLM put the entity name there, as in
        #    the DLQ rows of 2026-09-30. Treat a non-container value as a single
        #    entity and let the normaliser attach the empty required field.
        for cls_key in _CLASS_NAME_KEYS:
            if cls_key not in payload:
                continue
            val = payload[cls_key]
            if isinstance(val, list):
                return done(val)
            if isinstance(val, (dict, str, set, frozenset, tuple)):
                # A bare string is the entity name the LLM put under the class
                # name; a one-element set appears in the 2026-09-30 DLQ. Both
                # are recoverable, so hand them to the normaliser as a single
                # item rather than dropping them.
                return done(list(val) if isinstance(val, (set, frozenset, tuple)) else [val])

        # 7. A single entity dict.
        if any(marker in payload for marker in _SINGLE_ENTITY_MARKERS):
            return done([payload])

        # 8. Nothing recognisable. For a JSON-Schema echo ($defs / type /
        #    properties) there is genuinely no entity data here, so an empty
        #    list is the honest repair.
        return {default_key: []}
