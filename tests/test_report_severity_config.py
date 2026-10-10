"""Regression tests for report-sink severity configuration.

P1 (2026-10-10): ``WebhookSinkConfig`` had no ``min_severity`` field while
``config.example.yaml`` *and* the deployed ``config.yaml`` both declare one.
pydantic drops unknown keys without a warning, so an operator setting
``report.sinks.webhook.min_severity: "error"`` silently got the report-level
default instead. These tests pin the field so the key can never silently
vanish again.

Each direction is asserted separately: a single assertion that inverts when
the fix lands would be a check wired to the bug rather than to the contract.
"""

from __future__ import annotations

import pytest

from janus_graph.config import (
    CLISinkConfig,
    JanusSettings,
    ReportConfig,
    TelegramSinkConfig,
    WebhookSinkConfig,
)


def test_webhook_min_severity_is_honoured() -> None:
    """The documented key must reach the model instead of being dropped."""
    cfg = WebhookSinkConfig(**{"min_severity": "error"})
    assert cfg.min_severity == "error"


def test_webhook_min_severity_default() -> None:
    """Unset key falls back to "info", matching the Telegram/Pipe sinks."""
    assert WebhookSinkConfig().min_severity == "info"


@pytest.mark.parametrize(
    "cls,expected_default",
    [
        (CLISinkConfig, "warning"),
        (TelegramSinkConfig, "info"),
        (WebhookSinkConfig, "info"),
    ],
)
def test_every_severity_capable_sink_exposes_the_field(cls: type, expected_default: str) -> None:
    """Every sink the dispatcher reads `min_severity` from must declare it.

    dispatcher.py uses `getattr(sink_cfg, "min_severity", fallback)`; a sink
    missing the field is exactly how this bug hid. Guards the whole family,
    not just webhook.
    """
    assert hasattr(cls(), "min_severity")
    assert getattr(cls(), "min_severity") == expected_default


def test_webhook_min_severity_survives_full_settings_load() -> None:
    """End-to-end: the value must survive YAML-shaped load, not just kwargs.

    Unit-level kwargs prove the field exists; this proves the nested path
    ReportConfig -> ReportSinksConfig -> WebhookSinkConfig actually delivers
    it, which is the path config.yaml takes.
    """
    cfg = ReportConfig(
        min_severity="info",
        sinks={
            "webhook": {
                "enabled": True,
                "url": "https://example.invalid/hook",
                "min_severity": "error",
            }
        },
    )
    assert cfg.sinks.webhook.enabled is True
    assert cfg.sinks.webhook.min_severity == "error"


def test_unknown_keys_still_tolerated() -> None:
    """Adding the field must not have switched the model to extra="forbid".

    Pins that the fix is additive: an unrelated typo elsewhere in config.yaml
    keeps its existing (tolerated) behaviour rather than newly raising.
    """
    cfg = WebhookSinkConfig(**{"not_a_real_option": 1})
    assert not hasattr(cfg, "not_a_real_option")


def test_janus_settings_report_block_is_valid() -> None:
    """The live config shape must still validate after the field addition."""
    settings = JanusSettings(
        report={
            "min_severity": "info",
            "sinks": {"webhook": {"enabled": False, "min_severity": "error"}},
        }
    )
    assert settings.report.sinks.webhook.min_severity == "error"
