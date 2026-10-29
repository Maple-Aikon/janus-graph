"""Regression: the ``queue_status`` docstring must not promise a parameter that does not exist.

Measured bug (2026-10-29): the ``queue_status`` docstring said "Pass
``open_only=False`` to get the old behaviour", but the tool signature has
NO ``open_only`` parameter — the open-only listing is hard-coded
(``open_only=True`` at the call site). A caller reading the docstring would
try to pass the argument and get a validation error, or worse, assume a
recovered-rows view exists when it does not.

Two further traps in the old wording, both fixed here:

1. ``open_only=False`` at the QUEUE layer does NOT mean "the old
   behaviour". It means ``recovered_at IS NOT NULL`` — recovered rows ONLY.
   The pre-2026-10-16 behaviour was the MIXED set, which is
   ``open_only=None``. So the old sentence pointed at a behaviour that
   does not exist at any layer.
2. The "same population as ``stats.dlq``" claim is TRUE and must stay
   true: ``get_stats()['dlq']`` counts ``recovered_at IS NULL``, and the
   tool hard-codes ``open_only=True``. This test pins that so a future
   change to either side cannot silently re-split the population.

The guard is a DOCSTRING test on purpose: the parameter set is asserted
separately, so a future PR that really adds ``open_only`` will fail the
parameter assertion and be forced to update this file deliberately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from janus_graph.config import JanusSettings
from janus_graph.mcp.server import create_mcp_server


@pytest.fixture
def cfg(temp_dir: Path) -> JanusSettings:
    c = JanusSettings()
    c.pipeline.queue_db_path = str(temp_dir / "episodes.db")
    return c


def _schema_of(tool) -> dict:
    """Return the tool's input schema, whatever the SDK calls it.

    The installed MCP SDK (``mcp_types._types.Tool``) exposes the schema as
    ``input_schema``; the wire format name is ``inputSchema``. Looking the
    attribute up by name keeps this test from breaking on either.
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    assert schema is not None, "tool exposes no input schema attribute"
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump()
    return schema


def _queue_status_tool(tools) -> object:
    return next(t for t in tools if t.name == "queue_status")


@pytest.mark.asyncio
async def test_queue_status_has_no_open_only_parameter(cfg: JanusSettings):
    """The docstring must not advertise a parameter the signature lacks."""
    mcp = create_mcp_server(cfg)
    tools = await mcp.list_tools()
    tool = _queue_status_tool(tools)

    param_names = set(_schema_of(tool).get("properties", {}))

    assert "open_only" not in param_names, (
        "queue_status gained an open_only parameter — update "
        "test_queue_status_docstring_does_not_promise_a_missing_parameter "
        "and the docstring together."
    )


@pytest.mark.asyncio
async def test_queue_status_docstring_does_not_promise_a_missing_parameter(cfg: JanusSettings):
    """A docstring that says "Pass open_only=False" is a lie to the caller."""
    mcp = create_mcp_server(cfg)
    tools = await mcp.list_tools()
    tool = _queue_status_tool(tools)

    doc = tool.description or ""

    assert "Pass ``open_only=False``" not in doc
    # The corrected wording names the parameter only to say it is absent.
    assert "NO ``open_only`` parameter" in doc


@pytest.mark.asyncio
async def test_queue_status_docstring_still_claims_the_shared_population(cfg: JanusSettings):
    """The TRUE part of the old docstring must not be lost in the fix.

    ``stats.dlq`` and ``dlq_recent`` describe the same set of rows. The
    docstring is the only place an operator is told so, so the claim is
    pinned here alongside the population assertion in
    ``test_dlq_recent_shows_only_open_rows``.
    """
    mcp = create_mcp_server(cfg)
    tools = await mcp.list_tools()
    tool = _queue_status_tool(tools)

    doc = tool.description or ""

    assert "same population ``stats.dlq`` counts" in doc
