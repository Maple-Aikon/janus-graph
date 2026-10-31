"""Regression tests for SchemaRepairingLLMClient handling of non-dict payloads.

Context (2026-10-30, 3 open DLQ rows):
    "TypeError: graphiti_core.prompts.dedupe_nodes.NodeResolutions()
     argument after ** must be a mapping, not list"

Root cause:
    ``SchemaRepairingLLMClient.generate_response`` returned early whenever the
    inner client produced a non-dict payload::

        if response_model is None or not isinstance(response, dict):
            return response

    Every heuristic rule already knows how to repair a bare list
    (``{"entity_resolutions": payload}`` / ``{"extracted_entities": payload}``),
    but the early return meant ``find_rule`` was never reached for list
    payloads. The raw list was then handed to ``response_model(**payload)``
    at the Graphiti call site -> TypeError (not ValidationError), so the
    repair path was never entered and the quirks log stayed silent.

Fix:
    let non-dict payloads continue into the repair path; the rule registry
    decides whether it can repair them.
"""

import json

import pytest

from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.extract_nodes import ExtractedEntities

from janus_graph.heuristics.repairing_client import SchemaRepairingLLMClient


class _StubInner:
    """Minimal LLMClient stand-in that returns a canned payload."""

    def __init__(self, payload):
        self.payload = payload
        self.config = None
        self.model = "stub"
        self.small_model = "stub"
        self.temperature = 0.0
        self.max_tokens = 100

    async def _generate_response(
        self, messages, response_model=None, max_tokens=None, model_size=None
    ):
        return self.payload

    async def generate_response(self, messages, response_model=None, **kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_repairing_client_repairs_bare_list_payload(tmp_path):
    """A list payload must be repaired, not returned verbatim."""
    quirks = tmp_path / "quirks.jsonl"

    # NodeResolutions: bare list of resolution objects
    inner = _StubInner([{"id": 0, "name": "Assistant", "duplicate_candidate_id": -1}])
    client = SchemaRepairingLLMClient(inner=inner, quirks_log_path=str(quirks))
    out = await client.generate_response(messages=[], response_model=NodeResolutions)
    assert isinstance(out, dict), "expected repaired dict, got %s" % type(out)
    assert "entity_resolutions" in out
    NodeResolutions(**out)  # must actually parse

    # ExtractedEntities: bare list of entity objects
    inner2 = _StubInner([{"name": "Assistant", "entity_type_id": 0}])
    client2 = SchemaRepairingLLMClient(inner=inner2, quirks_log_path=str(quirks))
    out2 = await client2.generate_response(messages=[], response_model=ExtractedEntities)
    assert isinstance(out2, dict), "expected repaired dict, got %s" % type(out2)
    assert "extracted_entities" in out2
    ExtractedEntities(**out2)  # must actually parse

    # The repair must be visible in the quirks log (no silent swallow).
    assert quirks.exists()
    lines = [ln for ln in quirks.read_text().strip().splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["event_type"] == "repair"


@pytest.mark.asyncio
async def test_repairing_client_still_passes_through_valid_dict(tmp_path):
    """Valid dict payloads must pass through untouched (no regression)."""
    quirks = tmp_path / "quirks.jsonl"
    payload = {"entity_resolutions": [{"id": 0, "name": "A", "duplicate_candidate_id": -1}]}
    inner = _StubInner(payload)
    client = SchemaRepairingLLMClient(inner=inner, quirks_log_path=str(quirks))
    out = await client.generate_response(messages=[], response_model=NodeResolutions)
    assert out == payload
    assert not quirks.exists() or quirks.read_text().strip() == ""
