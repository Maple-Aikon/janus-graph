"""Tests for report dispatcher and sinks."""

import json
import pytest
from janus_graph.report.models import ReportEvent, ReportSeverity
from janus_graph.report.dispatcher import ReportDispatcher
from janus_graph.report.sinks.file import FileSink


@pytest.mark.asyncio
async def test_file_sink(temp_dir):
    report_file = temp_dir / "test_report.jsonl"
    sink = FileSink(str(report_file))

    event = ReportEvent(
        kind="test_event",
        severity=ReportSeverity.WARNING,
        summary="Test alert message",
        details={"code": 404},
    )

    await sink.emit(event)
    assert report_file.exists()

    lines = report_file.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["kind"] == "test_event"
    assert data["severity"] == "warning"
    assert data["summary"] == "Test alert message"
    assert data["details"]["code"] == 404


@pytest.mark.asyncio
async def test_report_dispatcher(temp_dir):
    report_file = temp_dir / "dispatch_report.jsonl"
    sink = FileSink(str(report_file))
    dispatcher = ReportDispatcher(sinks=[sink])

    event = ReportEvent(
        kind="batch_sweep",
        severity=ReportSeverity.INFO,
        summary="Sweep completed",
    )
    await dispatcher.emit(event)

    assert report_file.exists()
