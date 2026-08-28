"""Telegram notification sink."""

from __future__ import annotations

import httpx
from ..models import ReportEvent
from . import BaseSink


class TelegramSink(BaseSink):
    """Sends high-severity alerts to a Telegram chat."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def emit(self, event: ReportEvent) -> None:
        if not self.bot_token or not self.chat_id:
            return
        text = f"🚨 *Janus-Graph Alert* [{event.severity.value.upper()}]\n*{event.kind}*: {event.summary}"
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"})
        except Exception:
            pass
