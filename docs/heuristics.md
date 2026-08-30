# Janus-Graph LLM Schema-Repairing Heuristics 🛡️

When autonomous agents extract entities and relationships from conversational episodes, LLMs (especially local small models or LiteLLM gateways configured in `json_object` mode) frequently output JSON structures that deviate from Graphiti's strict Pydantic schemas (e.g. omitting default empty lists, stringifying lists of IDs, or generating nulls instead of arrays).

Without mitigation, these payload discrepancies trigger Pydantic `ValidationError` exceptions, immediately terminating processing and sending valid memory episodes to the Dead-Letter Queue (DLQ).

**Janus-Graph Heuristics Subsystem** intercepts these validation failures, repairs payloads in-memory via deterministic rules, re-validates the repaired dictionary, and completes extraction without re-querying the LLM or burning additional token budget.

---

## Subsystem Architecture

```text
+─────────────────────────────────────────────────────────────────────────────+
|                             Graphiti Extraction                             |
+─────────────────────────────────────────────────────────────────────────────+\n                                       │
                        LLMClient.generate_response()
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                        SchemaRepairingLLMClient                             |
|                                                                             |
|   1. Invokes inner LLM Client (OpenAIGenericClient, Anthropic, etc.)        |
|   2. Attempts Pydantic schema validation: response_model(**response)        |
|   3. If ValidationError:                                                    |
|      ├── Extracts raw input payload from err.errors()[i]["input"]           |
|      ├── Dispatches payload to HeuristicRegistry matching schema signature       |
|      ├── Applies deterministic transformations in-memory                    |
|      ├── Logs anomaly signature to QuirksLogger (JSONL)                     |
|      └── Re-validates repaired payload and returns clean instance           |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## Built-in Repair Rules

All rules inherit from `HeuristicRule` (`janus_graph.heuristics.rules.base.HeuristicRule`), registered into a `HeuristicRegistry` instance (`janus_graph.heuristics.registry.HeuristicRegistry`):

### 1. `EdgeDuplicateRule` (`rules/edge_duplicate.py`)
- **Target Schema**: `EdgeDuplicate`
- **Problem**: Models omit `duplicate_facts` or `contradicted_facts` fields, or return single integers/strings instead of lists.
- **Repair Actions**:
  - Normalizes `duplicate_facts` to `list[int]` (converts `None` -> `[]`, `"1, 2"` -> `[1, 2]`).
  - Normalizes `contradicted_facts` to `list[int]`.

### 2. `ExtractedEntitiesRule` (`rules/extracted_entities.py`)
- **Target Schema**: `ExtractedEntities`
- **Problem**: Models return a flat list or key `"items"` instead of `{ "entities": [...] }`, or individual entities miss `name` / `entity_type`.
- **Repair Actions**:
  - Wraps root arrays or dicts with missing root keys into `{ "entities": [...] }`.
  - Enforces `name` string coercion and defaults missing `entity_type` to `"Concept"`.

### 3. `ExtractedEdgesRule` (`rules/extracted_edges.py`)
- **Target Schema**: `ExtractedEdges`
- **Problem**: Edge objects missing required temporal attributes, source/target identifiers, or returning string boolean values.
- **Repair Actions**:
  - Normalizes `source_node_name`, `target_node_name`, `relation_type`.
  - Coerces `fact` and `episodes` list elements.

### 4. `NodeResolutionsRule` (`rules/node_resolutions.py`)
- **Target Schema**: `NodeResolutions`
- **Problem**: Missing `resolutions` list wrapper or malformed cluster mapping dictionaries.
- **Repair Actions**:
  - Reconstructs resolution maps and maps string cluster IDs to lists.

---

## Authoring Custom Repair Rules

You can implement custom repair rules by subclassing `HeuristicRule` and registering them with a `HeuristicRegistry` instance:

```python
from typing import Any, Dict
from janus_graph.heuristics.rules.base import HeuristicRule
from janus_graph.heuristics.registry import HeuristicRegistry

class CustomMemoryRule(HeuristicRule):
    @property
    def name(self) -> str:
        return "custom_memory_repair"

    @property
    def target_model(self) -> str:
        return "CustomSchema"

    @property
    def priority(self) -> int:
        return 50  # Lower number = higher priority

    def can_repair(self, schema_name: str, payload: Dict[str, Any], error: Any = None) -> bool:
        return schema_name == self.target_model or "custom_field" in payload

    def repair(self, schema_name: str, payload: Dict[str, Any], error: Any = None) -> Dict[str, Any]:
        repaired = dict(payload)
        if "tags" not in repaired or repaired["tags"] is None:
            repaired["tags"] = []
        elif isinstance(repaired["tags"], str):
            repaired["tags"] = [t.strip() for t in repaired["tags"].split(",")]
        return repaired

# Register the rule via an instantiated registry
registry = HeuristicRegistry(active_rules=["custom_memory_repair"])
registry.register(CustomMemoryRule())
```

---

## Telemetry & Quirks Logger

When a rule is triggered, the anomaly is recorded to `${DATA_DIR}/logs/llm_schema_quirks.jsonl`:

```json
{
  "timestamp": "2026-08-28T22:30:15Z",
  "model": "gpt-4o-mini",
  "schema": "EdgeDuplicate",
  "rule_applied": "edge_duplicate_repair",
  "error_fields": ["duplicate_facts", "contradicted_facts"],
  "raw_input_snippet": "{\"reasoning\": \"Facts are complementary\"}"
}
```

This structured audit log allows engineering teams to continuously identify model-specific formatting quirks and refine prompts or add heuristic rules without data loss.
