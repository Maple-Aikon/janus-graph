"""Telegram notification sink for alerts and sweep reports."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional
import urllib.request
from ..models import ReportEvent
from . import BaseSink

logger = logging.getLogger("janus_graph.report.telegram")


class TelegramSink(BaseSink):
    """Delivers formatted report notifications to Telegram channels or chats."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.backend = backend or os.environ.get("TELEGRAM_BACKEND", "api" if self.bot_token else "log")

    @property
    def name(self) -> str:
        return "telegram"

    def _format_message(self, event: ReportEvent) -> str:
        icon = {
            "debug": "🔍",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(event.severity.value, "ℹ️")
        lines = [
            f"{icon} <b>Janus-Graph Alert</b> [{event.severity.value.upper()}]",
            f"<b>Kind:</b> <code>{event.kind}</code>",
            f"<b>Summary:</b> {event.summary}",
        ]
        if event.details:
            lines.append(f"<pre>{json.dumps(event.details, indent=2, default=str)[:1000]}</pre>")
        return "\n".join(lines)

    async def emit(self, event: ReportEvent) -> None:
        if self.backend == "log" or not self.bot_token or not self.chat_id:
            logger.info("TelegramSink [log-mode]: %s (%s)", event.summary, event.kind)
            return

        text = self._format_message(event)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        def _send():
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return resp.read()

        try:
            await asyncio.to_thread(_send)
        except Exception as err:
            logger.warning("Failed to send Telegram notification: %s", err)
