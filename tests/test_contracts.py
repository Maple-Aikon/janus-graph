"""Tests for Phase 0.5 interface contract freeze."""

import pytest
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from janus_graph.core.contracts import (
    Settings,
    PipelineSettings,
    QueueSettings,
    DreamSettings,
    ReportSettings,
    CliReportSettings,
    TelegramReportSettings,
    WebhookReportSettings,
    LLMSettings,
    EmbeddingSettings,
    DatabaseSettings,
    PathsSettings,
    RepairEvent,
    DreamEvent,
    QuirksLogger,
    RepairRule,
    ReportSink,
)
from janus_graph.heuristics.quirks_logger import QuirksLogger as ImplQuirksLogger
from janus_graph.heuristics.rules.base import HeuristicRule
from janus_graph.heuristics.rules.edge_duplicate import EdgeDuplicateRule
from janus_graph.heuristics.rules.extracted_edges import ExtractedEdgesRule
from janus_graph.heuristics.rules.extracted_entities import ExtractedEntitiesRule
from janus_graph.heuristics.rules.node_resolutions import NodeResolutionsRule
from janus_graph.report.dispatcher import ReportDispatcher


def test_settings_immutability():
    """Verify that all settings dataclasses are frozen and immutable."""
    settings = Settings()
    assert is_dataclass(settings)

    with pytest.raises(FrozenInstanceError):
        settings.database = DatabaseSettings(port=9999)  # type: ignore

    with pytest.raises(FrozenInstanceError):
        settings.pipeline.queue.dual_run_enabled = True  # type: ignore

    with pytest.raises(FrozenInstanceError):
        settings.report.cli.pipe_user = "root"  # type: ignore


def test_event_dataclasses_frozen():
    """Verify that event dataclasses are frozen."""
    repair_event = RepairEvent(
        rule_name="edge_duplicate",
        target_model="EdgeDuplicate",
        episode_id="ep-123",
        payload_in={},
        payload_out={"duplicate_facts": []},
        success=True,
        duration_ms=1.5,
    )
    with pytest.raises(FrozenInstanceError):
        repair_event.success = False  # type: ignore

    dream_event = DreamEvent(
        phase="dedup",
        nodes_before=100,
        nodes_after=90,
        duration_ms=45.0,
        success=True,
    )
    with pytest.raises(FrozenInstanceError):
        dream_event.nodes_after = 80  # type: ignore


def test_protocols_runtime_checkable():
    """Verify runtime Protocol compliance across implementations."""
    # QuirksLogger
    logger = ImplQuirksLogger()
    assert isinstance(logger, QuirksLogger)

    # RepairRules
    for rule in (
        EdgeDuplicateRule(),
        ExtractedEdgesRule(),
        ExtractedEntitiesRule(),
        NodeResolutionsRule(),
    ):
        assert isinstance(rule, RepairRule)
        assert hasattr(rule, "name")
        assert hasattr(rule, "target_model")
        assert hasattr(rule, "priority")
        assert callable(rule.can_repair)
        assert callable(rule.repair)


def test_contracts_field_snapshot():
    """Snapshot test ensuring contract fields remain strictly stable."""
    queue = QueueSettings()
    assert hasattr(queue, "sqlite_busy_timeout_ms")
    assert hasattr(queue, "dual_run_busy_timeout_ms")
    assert hasattr(queue, "dual_run_enabled")
    assert hasattr(queue, "max_retries")

    report = ReportSettings()
    assert hasattr(report, "sinks")
    assert hasattr(report, "file_path")
    assert hasattr(report, "cli")
    assert hasattr(report, "telegram")
    assert hasattr(report, "webhook")
