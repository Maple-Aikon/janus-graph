"""CLI terminal stdout report sink."""

from __future__ import annotations

import sys
from ..models import ReportEvent
from . import BaseSink


class CLISink(BaseSink):
    """Outputs events to console."""

    def __init__(self, format_type: str = "pretty"):
        self.format_type = format_type

    async def emit(self, event: ReportEvent) -> None:
        msg = f"[{event.severity.value.upper()}] [{event.kind}] {event.summary}"
        print(msg, file=sys.stderr if event.severity.value in ("error", "critical") else sys.stdout)
