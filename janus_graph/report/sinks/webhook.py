"""Generic HTTP Webhook notification sink."""

from __future__ import annotations

import hmac
import hashlib
import json
import httpx
from ..models import ReportEvent
from . import BaseSink


class WebhookSink(BaseSink):
    """Sends JSON events to a configured webhook endpoint with HMAC signature."""

    def __init__(self, url: str, secret_token: str = "", secret_header: str = "X-Janus-Signature"):
        self.url = url
        self.secret_token = secret_token
        self.secret_header = secret_header

    async def emit(self, event: ReportEvent) -> None:
        if not self.url:
            return
        payload = {
            "timestamp": event.timestamp,
            "kind": event.kind,
            "severity": event.severity.value,
            "summary": event.summary,
            "details": event.details,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.secret_token:
            sig = hmac.new(self.secret_token.encode("utf-8"), body, hashlib.sha256).hexdigest()
            headers[self.secret_header] = f"sha256={sig}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self.url, content=body, headers=headers)
        except Exception:
            pass
