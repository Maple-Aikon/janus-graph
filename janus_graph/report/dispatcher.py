"""Report dispatcher routing events to configured sinks."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Union
from ..config import JanusSettings, load_config
from ..core.contracts import Settings
from .models import ReportEvent, ReportSeverity
from .sinks import BaseSink
from .sinks.cli import CLISink
from .sinks.file import FileSink
from .sinks.telegram import TelegramSink
from .sinks.webhook import WebhookSink

logger = logging.getLogger("janus_graph.report.dispatcher")


class ReportDispatcher:
    """Dispatches report events across multiple registered sinks concurrently."""

    def __init__(self, sinks: Optional[List[BaseSink]] = None):
        self.sinks = sinks or []

    @classmethod
    def from_settings(cls, settings: Optional[Union[JanusSettings, Settings]] = None) -> ReportDispatcher:
        cfg = settings or load_config()
        sinks: List[BaseSink] = []

        # Handle JanusSettings (pydantic)
        if isinstance(cfg, JanusSettings):
            if cfg.report.sinks.file.enabled:
                sinks.append(
                    FileSink(
                        path=cfg.report.sinks.file.path,
                        rotation_max_bytes=cfg.report.sinks.file.rotation_max_bytes,
                    )
                )
            if cfg.report.sinks.cli.enabled:
                sinks.append(CLISink(format_type=cfg.report.sinks.cli.format, min_severity=getattr(cfg.report.sinks.cli, "min_severity", cfg.report.min_severity)))
            if cfg.report.sinks.telegram.enabled:
                sinks.append(
                    TelegramSink(
                        bot_token=cfg.report.sinks.telegram.bot_token,
                        chat_id=cfg.report.sinks.telegram.chat_id,
                        min_severity=getattr(cfg.report.sinks.telegram, "min_severity", cfg.report.min_severity),
                    )
                )
            if cfg.report.sinks.webhook.enabled:
                sinks.append(
                    WebhookSink(
                        url=cfg.report.sinks.webhook.url,
                        secret_token=cfg.report.sinks.webhook.secret_token,
                        secret_header=cfg.report.sinks.webhook.secret_header,
                        min_severity=getattr(cfg.report.sinks.webhook, "min_severity", cfg.report.min_severity),
                    )
                )
        elif isinstance(cfg, Settings):
            # Handle frozen Settings dataclass
            if "file" in cfg.report.sinks:
                sinks.append(FileSink(path=str(cfg.report.file_path)))
            if "cli" in cfg.report.sinks:
                sinks.append(CLISink(subprocess_method=cfg.report.cli.pipe_subprocess_method))
            if cfg.report.telegram.enabled or "telegram" in cfg.report.sinks:
                sinks.append(TelegramSink(bot_token=cfg.report.telegram.bot_token, chat_id=cfg.report.telegram.chat_id))
            if cfg.report.webhook.enabled or "webhook" in cfg.report.sinks:
                sinks.append(WebhookSink(url=cfg.report.webhook.url, secret_token=cfg.report.webhook.secret))
        return cls(sinks=sinks)

    async def emit(self, event: ReportEvent) -> None:
        """Emit a typed event to all sinks concurrently with error isolation."""
        if not self.sinks:
            return
        await asyncio.gather(*(sink.emit(event) for sink in self.sinks), return_exceptions=True)

    async def emit_quick(
        self,
        kind: str,
        summary: str,
        severity: ReportSeverity = ReportSeverity.INFO,
        details: Optional[dict] = None,
    ) -> None:
        """Helper to quickly construct and emit an event."""
        event = ReportEvent(
            kind=kind,
            severity=severity,
            summary=summary,
            details=details or {},
        )
        await self.emit(event)
