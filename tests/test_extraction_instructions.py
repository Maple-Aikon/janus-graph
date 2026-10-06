"""Regression tests for the L1 extraction-instruction plumbing (fc3e4e8).

`graphiti.custom_extraction_instructions` is read by EpisodeWorker and
forwarded to ``add_episode`` as a kwarg. Three behaviours are pinned here:

1. configured      -> the kwarg reaches ``add_episode`` with the exact string
2. not configured  -> the kwarg is ABSENT from the call, not present-and-None
3. no graphiti section -> ingestion still proceeds (getattr chain, not AttributeError)

Cases 2 and 3 are the load-bearing ones. The worker deliberately builds
``ingest_kwargs`` and only inserts the key when the value is truthy, because
passing ``custom_extraction_instructions=None`` changes the ``add_episode``
call shape. A regression to an always-pass form would silently alter the call
for every deployment that never sets the field.

HERMETICITY (learned the hard way, 2026-10-06): ``JanusSettings()`` resolves its
YAML through ``_resolve_default_yaml_file()``, which prefers
``$JANUS_GRAPH_HOME/config.yaml``. The daemon sets that variable, so a bare
``JanusSettings()`` inside the suite silently inherits the operator's live
prompt - and the "unset" case below would fail on the daemon's environment
while passing in a developer shell. Every settings object here therefore pins
the graphiti section explicitly as an init kwarg, which outranks the YAML
source in pydantic-settings.
"""

from unittest.mock import AsyncMock, patch

import pytest

from janus_graph.config import GraphitiConfig, JanusSettings
from janus_graph.pipeline.worker import EpisodeWorker


def _settings(custom_instructions=None):
    """Build settings with an EXPLICIT graphiti section, ignoring any config.yaml.

    Passing the section as an init kwarg wins over every other settings source,
    so the returned object is identical whether or not JANUS_GRAPH_HOME points
    at a live deployment.
    """
    return JanusSettings(
        graphiti=GraphitiConfig(custom_extraction_instructions=custom_instructions)
    )


async def _run_one_episode(queue, settings):
    """Enqueue one episode, process it, return the mock graphiti client."""
    ep_id = await queue.enqueue("A fact worth remembering", group_id="grp_forward")
    records = await queue.claim_next_batch(1)
    assert len(records) == 1

    mock_client = AsyncMock()
    mock_client.add_episode = AsyncMock(return_value=None)

    worker = EpisodeWorker(queue, settings)
    with patch(
        "janus_graph.core.instance.create_graphiti_instance", return_value=mock_client
    ):
        ok = await worker.process_record(records[0])

    assert ok is True, "episode should ingest successfully"
    return ep_id, mock_client


@pytest.mark.asyncio
async def test_custom_extraction_instructions_forwarded_to_add_episode(temp_queue):
    """A configured instruction string must arrive at add_episode verbatim."""
    _ep_id, client = await _run_one_episode(
        temp_queue, _settings("NEVER extract file paths.")
    )

    assert client.add_episode.await_count == 1
    kwargs = client.add_episode.await_args.kwargs
    assert "custom_extraction_instructions" in kwargs, (
        "configured instruction did not reach add_episode; "
        f"keys seen: {sorted(kwargs)}"
    )
    assert kwargs["custom_extraction_instructions"] == "NEVER extract file paths."
    # the rest of the call must be untouched by the new kwarg
    assert kwargs["episode_body"] == "A fact worth remembering"
    assert kwargs["group_id"] == "grp_forward"


@pytest.mark.asyncio
async def test_custom_extraction_instructions_absent_when_unset(temp_queue):
    """Unset must mean ABSENT, not present-and-None.

    The worker builds ingest_kwargs and only inserts the key when the value is
    truthy. This test fails if that guard is removed and the kwarg is passed
    unconditionally.
    """
    settings = _settings(None)
    assert settings.graphiti.custom_extraction_instructions is None

    _ep_id, client = await _run_one_episode(temp_queue, settings)

    kwargs = client.add_episode.await_args.kwargs
    assert "custom_extraction_instructions" not in kwargs, (
        "unset instruction must not appear in the add_episode call at all; "
        f"keys seen: {sorted(kwargs)}"
    )


@pytest.mark.asyncio
async def test_custom_extraction_instructions_ignores_empty_string(temp_queue):
    """An empty string is falsy, so it must behave like 'not configured'.

    This is the boundary of the ``if custom_instructions:`` guard. An empty
    prompt is a config mistake; forwarding it would change the call shape for
    no benefit.
    """
    settings = _settings("")
    assert settings.graphiti.custom_extraction_instructions == ""

    _ep_id, client = await _run_one_episode(temp_queue, settings)

    kwargs = client.add_episode.await_args.kwargs
    assert "custom_extraction_instructions" not in kwargs, (
        "empty-string instruction should be treated as unset; "
        f"keys seen: {sorted(kwargs)}"
    )


@pytest.mark.asyncio
async def test_worker_survives_settings_without_graphiti_section(temp_queue):
    """Settings objects with no graphiti section must not raise AttributeError.

    ``Settings(report=...)`` has no graphiti attribute at all. The worker
    reaches for the field through a getattr chain precisely so ingestion still
    proceeds. A direct ``self.settings.graphiti.custom_...`` would raise
    before add_episode is ever called.
    """
    from janus_graph.core.contracts import ReportSettings, Settings

    settings = Settings(report=ReportSettings(sinks=()))
    assert not hasattr(settings, "graphiti"), "fixture premise changed"

    _ep_id, client = await _run_one_episode(temp_queue, settings)

    kwargs = client.add_episode.await_args.kwargs
    assert "custom_extraction_instructions" not in kwargs
    assert kwargs["episode_body"] == "A fact worth remembering"