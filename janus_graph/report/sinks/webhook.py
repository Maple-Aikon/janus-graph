"""Webhook notification sink with HMAC signature verification support."""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import os
from typing import Optional
import urllib.request
from ..models import ReportEvent
from . import BaseSink

logger = logging.getLogger("janus_graph.report.webhook")


class WebhookSink(BaseSink):
    """Delivers report events via HTTP POST webhook with HMAC signing."""

    def __init__(
        self,
        url: Optional[str] = None,
        secret_token: Optional[str] = None,
        secret_header: str = "X-Janus-Signature",
        timeout_sec: float = 10.0,
    ):
        self.url = url or os.environ.get("WEBHOOK_URL")
        self.secret_token = secret_token or os.environ.get("WEBHOOK_SECRET")
        self.secret_header = secret_header
        self.timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "webhook"

    def _compute_signature(self, body_bytes: bytes) -> str:
        if not self.secret_token:
            return ""
        return hmac.new(self.secret_token.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()

    async def emit(self, event: ReportEvent) -> None:
        if not self.url:
            return

        payload_bytes = json.dumps(event.to_dict(), default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Janus-Graph-Reporter/1.0",
        }
        if self.secret_token:
            headers[self.secret_header] = f"sha256={self._compute_signature(payload_bytes)}"

        def _post():
            req = urllib.request.Request(
                self.url,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return resp.read()

        try:
            await asyncio.to_thread(_post)
        except Exception as err:
            logger.warning("Failed to send Webhook notification to %s: %s", self.url, err)
