"""Tests for heuristics rules and quirks logger."""

import json
from pathlib import Path

import pytest

from janus_graph.heuristics.quirks_logger import QuirksLogger
from janus_graph.heuristics.registry import HeuristicRegistry
from janus_graph.heuristics.rules.edge_duplicate import EdgeDuplicateRule
from janus_graph.heuristics.rules.extracted_edges import ExtractedEdgesRule
from janus_graph.heuristics.rules.extracted_entities import ExtractedEntitiesRule
from janus_graph.heuristics.rules.node_resolutions import NodeResolutionsRule

# ---------------------------------------------------------------------------
# Rule 1: EdgeDuplicateRule (Signatures 5 & variations)
# ---------------------------------------------------------------------------


def test_edge_duplicate_rule_basics():
    rule = EdgeDuplicateRule()
    assert rule.name == "edge_duplicate"
    assert rule.target_model == "EdgeDuplicate"
    assert rule.can_repair("EdgeDuplicate", {})

    # Corrupted payload missing list fields
    bad_payload = {"some_key": "val"}
    repaired = rule.repair("EdgeDuplicate", bad_payload)
    assert repaired == {"duplicate_facts": [], "contradicted_facts": []}

    # Payload with comma-separated string
    str_payload = {"duplicate_facts": "1, 2, 3", "contradicted_facts": None}
    repaired_str = rule.repair("EdgeDuplicate", str_payload)
    assert repaired_str == {"duplicate_facts": [1, 2, 3], "contradicted_facts": []}


def test_edge_duplicate_nested_properties():
    rule = EdgeDuplicateRule()
    nested = {"properties": {"duplicate_facts": [0, 2], "contradicted_facts": [1]}}
    repaired = rule.repair("EdgeDuplicate", nested)
    assert repaired == {"duplicate_facts": [0, 2], "contradicted_facts": [1]}


def test_edge_duplicate_key_aliases():
    rule = EdgeDuplicateRule()
    aliases = {"duplicate_indices": [1], "contradicted_indices": [2, 3]}
    repaired = rule.repair("EdgeDuplicate", aliases)
    assert repaired == {"duplicate_facts": [1], "contradicted_facts": [2, 3]}


# ---------------------------------------------------------------------------
# Rule 2: ExtractedEdgesRule (Signatures 1, 3)
# ---------------------------------------------------------------------------


def test_extracted_edges_rule():
    rule = ExtractedEdgesRule()
    assert rule.name == "extracted_edges"
    assert rule.can_repair("ExtractedEdges", {})

    # Signature 1: facts key alias
    repaired = rule.repair("ExtractedEdges", {"facts": [{"src": "a", "dst": "b"}]})
    assert repaired == {"edges": [{"src": "a", "dst": "b"}]}

    # Signature 3: Bare list
    repaired_list = rule.repair("ExtractedEdges", [{"src": "x"}])
    assert repaired_list == {"edges": [{"src": "x"}]}

    # Nested properties wrapper
    nested = {"properties": {"edges": [{"src": "p"}]}}
    repaired_nested = rule.repair("ExtractedEdges", nested)
    assert repaired_nested == {"edges": [{"src": "p"}]}


# ---------------------------------------------------------------------------
# Rule 3: ExtractedEntitiesRule (Signatures 2, 4)
# ---------------------------------------------------------------------------


def test_extracted_entities_rule():
    rule = ExtractedEntitiesRule()
    assert rule.name == "extracted_entities"
    assert rule.can_repair("ExtractedEntities", {})

    # Every repaired entity is normalised: entity_type_id and episode_indices
    # are required/ defaulted by graphiti_core, so the rule always fills them.
    def norm(name, **kw):
        out = {"name": name, "entity_type_id": 0, "episode_indices": [0]}
        out.update(kw)
        return out

    # Signature 2: entities key alias
    repaired = rule.repair("ExtractedEntities", {"entities": [{"name": "Maple"}]})
    assert repaired == {"extracted_entities": [norm("Maple")]}

    # Signature 4: Bare list
    repaired_list = rule.repair("ExtractedEntities", [{"name": "Alice"}, {"name": "Bob"}])
    assert repaired_list == {
        "extracted_entities": [norm("Alice"), norm("Bob")]
    }

    # Nested properties wrapper
    nested = {"properties": {"extracted_entities": [{"name": "Thúy Vi"}]}}
    repaired_nested = rule.repair("ExtractedEntities", nested)
    assert repaired_nested == {"extracted_entities": [norm("Thúy Vi")]}


def test_extracted_entities_preserves_string_entity_type():
    """A string ``entity_type`` label is kept, and the id defaults to 0."""
    rule = ExtractedEntitiesRule()
    repaired = rule.repair(
        "ExtractedEntities", {"entities": [{"name": "Maple", "entity_type": "person"}]}
    )
    got = repaired["extracted_entities"][0]
    assert got["entity_type"] == "person"
    assert got["entity_type_id"] == 0


def test_extracted_entities_bool_type_id_is_not_one():
    """bool is a subclass of int, so int(True) == 1 would misclassify."""
    rule = ExtractedEntitiesRule()
    repaired = rule.repair(
        "ExtractedEntities", [{"name": "X", "entity_type_id": True}]
    )
    assert repaired["extracted_entities"][0]["entity_type_id"] == 0


def test_extracted_entities_coerces_string_indices():
    rule = ExtractedEntitiesRule()
    repaired = rule.repair(
        "ExtractedEntities",
        [{"name": "X", "entity_type_id": 2, "episode_indices": "4"}],
    )
    assert repaired["extracted_entities"][0]["episode_indices"] == [4]


def test_extracted_entities_repairs_json_string_payload():
    rule = ExtractedEntitiesRule()
    repaired = rule.repair(
        "ExtractedEntities", '{"extracted_entities": [{"name": "E"}]}'
    )
    assert repaired == {
        "extracted_entities": [{"name": "E", "entity_type_id": 0, "episode_indices": [0]}]
    }


# ---------------------------------------------------------------------------
# SummarizedEntities (added 2026-10-02)
#
# Before this, SummarizedEntities had NO rule at all: HeuristicRegistry
# .find_rule returned None, repairing_client re-raised, and 21 DLQ rows sat
# unrepairable. Shapes below are taken from the real dead_letter rows.
# ---------------------------------------------------------------------------


def test_summarized_entities_is_claimed_by_the_rule():
    rule = ExtractedEntitiesRule()
    assert rule.can_repair("SummarizedEntities", {})
    assert "SummarizedEntities" in rule.target_schema_names
    assert rule.target_schema_names == ["ExtractedEntities", "SummarizedEntities"]


def test_registry_resolves_summarized_entities():
    registry = HeuristicRegistry()
    assert registry.find_rule("SummarizedEntities", {}, None) is not None
    assert registry.find_rule("NoSuchSchema", {}, None) is None


@pytest.mark.parametrize(
    "payload,expected",
    [
        # {"summaries": [...]} — already correct
        ({"summaries": [{"name": "A", "summary": "s"}]}, [{"name": "A", "summary": "s"}]),
        # alias
        ({"summarized_entities": [{"name": "B"}]}, [{"name": "B", "summary": ""}]),
        # properties wrapper (real DLQ shape)
        ({"properties": {"summaries": [{"name": "C"}]}}, [{"name": "C", "summary": ""}]),
        # pydantic hands us the *inner* entity, not the envelope (real DLQ)
        ({"name": "D"}, [{"name": "D", "summary": ""}]),
        # JSON-Schema array echo (real DLQ shape)
        ({"items": [{"name": "E"}], "type": "array"}, [{"name": "E", "summary": ""}]),
        # class-name wrapper, list form (real DLQ shape)
        ({"SummarizedEntity": [{"name": "F", "summary": "t"}]}, [{"name": "F", "summary": "t"}]),
        # class-name wrapper, bare string form (real DLQ shape)
        ({"SummarizedEntity": "MEMORY.md"}, [{"name": "<unnamed>", "summary": "MEMORY.md"}]),
        # one-element set under the class name (real DLQ shape, 2026-09-30)
        ({"SummarizedEntity": {"one item"}}, [{"name": "<unnamed>", "summary": "one item"}]),
        # bare list
        ([{"name": "G", "summary": "v"}], [{"name": "G", "summary": "v"}]),
        # bare string inside a list
        (["plain text"], [{"name": "<unnamed>", "summary": "plain text"}]),
        # raw JSON string
        ('{"summaries": [{"name": "H", "summary": "w"}]}', [{"name": "H", "summary": "w"}]),
    ],
)
def test_summarized_entities_repair_shapes(payload, expected):
    rule = ExtractedEntitiesRule()
    assert rule.repair("SummarizedEntities", payload) == {"summaries": expected}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        # A JSON-Schema echo carries no entity data; an empty list is the
        # honest repair. It must NOT be mistaken for a successful rescue.
        {"$defs": {"SummarizedEntity": {"type": "object"}}},
        {"type": "object", "properties": {}},
    ],
)
def test_summarized_entities_empty_only_when_no_data(payload):
    rule = ExtractedEntitiesRule()
    assert rule.repair("SummarizedEntities", payload) == {"summaries": []}


def test_summarized_entities_output_passes_real_schema():
    """Guards against a repair that validates against nothing."""
    extract_nodes = pytest.importorskip("graphiti_core.prompts.extract_nodes")
    rule = ExtractedEntitiesRule()
    for payload in (
        {"SummarizedEntity": "MEMORY.md"},
        {"SummarizedEntity": {"one item"}},
        {"items": [{"name": "E"}], "type": "array"},
        {"name": "D"},
    ):
        repaired = rule.repair("SummarizedEntities", payload)
        extract_nodes.SummarizedEntities.model_validate(repaired)


# ---------------------------------------------------------------------------
# Rule 4: NodeResolutionsRule (Signature 6)
# ---------------------------------------------------------------------------


def test_node_resolutions_rule():
    rule = NodeResolutionsRule()
    assert rule.name == "node_resolutions"
    assert rule.can_repair("NodeResolutions", {})

    # Signature 6: Bare resolution object
    single_res = {"id": 0, "name": "Alice", "duplicate_candidate_id": 0}
    repaired_single = rule.repair("NodeResolutions", single_res)
    assert repaired_single == {"entity_resolutions": [single_res]}

    # Bare list
    list_res = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}]
    repaired_list = rule.repair("NodeResolutions", list_res)
    assert repaired_list == {"entity_resolutions": list_res}


# ---------------------------------------------------------------------------
# Registry & Logger Tests
# ---------------------------------------------------------------------------


def test_registry_ordering_and_filtering():
    registry = HeuristicRegistry(active_rules=["edge_duplicate", "extracted_edges"])
    rules = registry.get_rules()
    assert len(rules) == 2
    assert rules[0].name == "edge_duplicate"
    assert rules[1].name == "extracted_edges"

    # EdgeDuplicate matching
    match = registry.find_rule("EdgeDuplicate", {})
    assert match is not None
    assert match.name == "edge_duplicate"

    # NodeResolutions not active
    match_disabled = registry.find_rule("NodeResolutions", {})
    assert match_disabled is None


def test_quirks_logger(temp_dir):
    log_file = temp_dir / "quirks.jsonl"
    logger = QuirksLogger(str(log_file))

    # Log repair
    logger.log_repair(
        rule_name="edge_duplicate",
        schema_name="EdgeDuplicate",
        original_payload={"bad": 1},
        repaired_payload={"duplicate_facts": []},
        error_message="Test validation error",
        episode_id="ep-101",
    )

    # Log quirk
    logger.log_schema_quirk(
        model_name="deepseek-chat",
        field="duplicate_facts",
        observed_type=type(None),
        expected_type=list,
        repair_applied=True,
        episode_id="ep-101",
    )

    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    repair_entry = json.loads(lines[0])
    assert repair_entry["event_type"] == "repair"
    assert repair_entry["episode_id"] == "ep-101"

    quirk_entry = json.loads(lines[1])
    assert quirk_entry["event_type"] == "schema_quirk"
    assert quirk_entry["repair_applied"] is True
