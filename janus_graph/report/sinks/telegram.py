"""Telegram direct bot-API sink for alerts and sweep reports.

Single backend: ``POST`` JSON to ``https://api.telegram.org/bot{bot_token}/sendMessage``
via ``urllib``. Requires ``bot_token`` + ``chat_id`` in config (or matching
env vars). For PicoClaw or other custom-routing consumers, prefer the
generic :class:`PipeSink` (``janus_graph.report.sinks.pipe``) and configure
it to call e.g. ``send_telegram.sh``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import Optional

from ..models import ReportEvent
from . import BaseSink

logger = logging.getLogger("janus_graph.report.telegram")


class TelegramSink(BaseSink):
    """Delivers formatted report notifications to a Telegram chat via bot API.

    Parameters
    ----------
    bot_token
        Telegram bot token. Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
    chat_id
        Telegram chat id. Falls back to ``TELEGRAM_CHAT_ID`` env var.
    min_severity
        Drop events below this severity (default ``"info"`` → no filtering).
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        min_severity: str = "info",
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.min_severity = min_severity

    @property
    def name(self) -> str:
        return "telegram"

    def _icon(self, severity_value: str) -> str:
        return {
            "debug": "🔍",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(severity_value, "ℹ️")

    def _format_message(self, event: ReportEvent) -> str:
        """Render an event into HTML for the bot-API path."""
        icon = self._icon(event.severity.value)
        lines = [
            f"{icon} <b>Janus-Graph Alert</b> [{event.severity.value.upper()}]",
            f"<b>Kind:</b> <code>{event.kind}</code>",
            f"<b>Summary:</b> {event.summary}",
        ]
        if event.details:
            lines.append(f"<pre>{json.dumps(event.details, indent=2, default=str)[:1000]}</pre>")
        return chr(10).join(lines)

    async def emit(self, event: ReportEvent) -> None:
        if not event.severity.is_at_least(self.min_severity):
            return
        if not self.bot_token or not self.chat_id:
            logger.warning(
                "TelegramSink: missing bot_token/chat_id — skipping emit "
                "(consider configuring a PipeSink instead for custom routing)"
            )
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
