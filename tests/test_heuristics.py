"""Tests for heuristics rules and quirks logger."""

import json
import pytest
from pathlib import Path
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

    # Signature 2: entities key alias
    repaired = rule.repair("ExtractedEntities", {"entities": [{"name": "Maple"}]})
    assert repaired == {"extracted_entities": [{"name": "Maple"}]}

    # Signature 4: Bare list
    repaired_list = rule.repair("ExtractedEntities", [{"name": "Alice"}, {"name": "Bob"}])
    assert repaired_list == {"extracted_entities": [{"name": "Alice"}, {"name": "Bob"}]}

    # Nested properties wrapper
    nested = {"properties": {"extracted_entities": [{"name": "Thúy Vi"}]}}
    repaired_nested = rule.repair("ExtractedEntities", nested)
    assert repaired_nested == {"extracted_entities": [{"name": "Thúy Vi"}]}


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
