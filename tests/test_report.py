"""Tests for report dispatcher and sinks."""

import io
import json
from unittest.mock import MagicMock, patch
import pytest
from janus_graph.report.models import ReportEvent, ReportSeverity
from janus_graph.report.dispatcher import ReportDispatcher
from janus_graph.report.sinks.file import FileSink
from janus_graph.report.sinks.cli import CLISink
from janus_graph.report.sinks.telegram import TelegramSink
from janus_graph.report.sinks.webhook import WebhookSink
from janus_graph.core.contracts import Settings, ReportSettings, CliReportSettings, TelegramReportSettings, WebhookReportSettings


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
async def test_file_sink_rotation(temp_dir):
    report_file = temp_dir / "rotating_report.jsonl"
    sink = FileSink(str(report_file), rotation_max_bytes=100)

    # Emit multiple events to exceed 100 bytes
    for i in range(5):
        await sink.emit(ReportEvent(kind="test", severity=ReportSeverity.INFO, summary=f"Event {i}" * 5))

    assert report_file.exists()
    rotated = temp_dir / "rotating_report.jsonl.1"
    assert rotated.exists()


@pytest.mark.asyncio
async def test_cli_sink():
    sink = CLISink(format_type="pretty")
    event = ReportEvent(kind="cli_test", severity=ReportSeverity.INFO, summary="Console message")
    await sink.emit(event)

    # Test render methods
    rendered_pretty = sink.render(event)
    assert "cli_test" in rendered_pretty
    assert "Console message" in rendered_pretty

    summary_sink = CLISink(format_type="summary")
    rendered_summary = summary_sink.render(event)
    assert rendered_summary == "[INFO] cli_test: Console message"

    json_sink = CLISink(format_type="json")
    rendered_json = json_sink.render(event)
    assert json.loads(rendered_json)["summary"] == "Console message"

    # Test strip ansi
    ansi_text = "\033[32mColored\033[0m Text"
    assert sink._strip_ansi(ansi_text) == "Colored Text"

    # Invalid method should raise
    with pytest.raises(ValueError):
        CLISink(subprocess_method="invalid_method")


@pytest.mark.asyncio
async def test_telegram_sink_log_mode():
    sink = TelegramSink(backend="log")
    event = ReportEvent(kind="tg_test", severity=ReportSeverity.CRITICAL, summary="Server down!")
    await sink.emit(event)
    assert sink.name == "telegram"


@pytest.mark.asyncio
async def test_telegram_sink_api_mode():
    sink = TelegramSink(bot_token="test_token", chat_id="12345", backend="api")
    event = ReportEvent(kind="tg_api", severity=ReportSeverity.ERROR, summary="Test error")
    
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"ok": true}'
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        await sink.emit(event)
        assert mock_urlopen.called


@pytest.mark.asyncio
async def test_webhook_sink():
    sink = WebhookSink(url=None, secret_token="my_secret")
    sig = sink._compute_signature(b'{"hello": "world"}')
    assert len(sig) == 64  # sha256 hex string
    assert sink.name == "webhook"


@pytest.mark.asyncio
async def test_webhook_sink_emit():
    sink = WebhookSink(url="http://example.com/webhook", secret_token="my_secret")
    event = ReportEvent(kind="webhook_test", severity=ReportSeverity.INFO, summary="Webhook payload")

    mock_response = MagicMock()
    mock_response.read.return_value = b'{"received": true}'
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        await sink.emit(event)
        assert mock_urlopen.called


@pytest.mark.asyncio
async def test_report_dispatcher(temp_dir):
    report_file = temp_dir / "dispatch_report.jsonl"
    sink = FileSink(str(report_file))
    dispatcher = ReportDispatcher(sinks=[sink, CLISink(), TelegramSink(backend="log")])

    await dispatcher.emit_quick(
        kind="batch_sweep",
        summary="Sweep completed successfully",
        severity=ReportSeverity.INFO,
        details={"processed": 10},
    )

    assert report_file.exists()
    data = json.loads(report_file.read_text().strip())
    assert data["summary"] == "Sweep completed successfully"


@pytest.mark.asyncio
async def test_report_dispatcher_error_isolation():
    async def _failing_emit(event):
        raise RuntimeError("Sink explosion")
    
    failing_sink = MagicMock()
    failing_sink.emit = _failing_emit
    
    working_sink = MagicMock()
    async def _ok(event): pass
    working_sink.emit = _ok

    dispatcher = ReportDispatcher(sinks=[failing_sink, working_sink])
    # Should not raise exception
    await dispatcher.emit_quick(kind="test", summary="Resilience check")


def test_report_dispatcher_from_settings(temp_dir):
    settings = Settings(
        report=ReportSettings(
            sinks=("file", "cli", "telegram", "webhook"),
            file_path=temp_dir / "contract_report.jsonl",
            cli=CliReportSettings(pipe_subprocess_method="sudo"),
            telegram=TelegramReportSettings(enabled=True, bot_token="tok", chat_id="chat"),
            webhook=WebhookReportSettings(enabled=True, url="http://hook.local", secret="sec"),
        )
    )
    dispatcher = ReportDispatcher.from_settings(settings)
    assert len(dispatcher.sinks) == 4
