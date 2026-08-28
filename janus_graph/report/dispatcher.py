"""Report dispatcher routing events to configured sinks."""

from __future__ import annotations

import asyncio
from typing import List, Optional
from ..config import JanusSettings, load_config
from .models import ReportEvent
from .sinks import BaseSink
from .sinks.cli import CLISink
from .sinks.file import FileSink
from .sinks.telegram import TelegramSink
from .sinks.webhook import WebhookSink


class ReportDispatcher:
    """Dispatches report events across multiple registered sinks."""

    def __init__(self, sinks: Optional[List[BaseSink]] = None):
        self.sinks = sinks or []

    @classmethod
    def from_settings(cls, settings: Optional[JanusSettings] = None) -> ReportDispatcher:
        cfg = settings or load_config()
        sinks: List[BaseSink] = []
        
        if cfg.report.sinks.file.enabled:
            sinks.append(
                FileSink(
                    path=cfg.report.sinks.file.path,
                    rotation_max_bytes=cfg.report.sinks.file.rotation_max_bytes,
                )
            )
        if cfg.report.sinks.cli.enabled:
            sinks.append(CLISink(format_type=cfg.report.sinks.cli.format))
        if cfg.report.sinks.telegram.enabled:
            sinks.append(
                TelegramSink(
                    bot_token=cfg.report.sinks.telegram.bot_token,
                    chat_id=cfg.report.sinks.telegram.chat_id,
                )
            )
        if cfg.report.sinks.webhook.enabled:
            sinks.append(
                WebhookSink(
                    url=cfg.report.sinks.webhook.url,
                    secret_token=cfg.report.sinks.webhook.secret_token,
                    secret_header=cfg.report.sinks.webhook.secret_header,
                )
            )
        return cls(sinks=sinks)

    async def emit(self, event: ReportEvent) -> None:
        """Emit an event to all sinks concurrently with error isolation."""
        if not self.sinks:
            return
        await asyncio.gather(*(sink.emit(event) for sink in self.sinks), return_exceptions=True)
