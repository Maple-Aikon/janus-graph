"""CLI terminal stdout report sink with secure formatting."""

from __future__ import annotations

import json
import re
import sys
from typing import Optional
from ..models import ReportEvent
from . import BaseSink

ALLOWED_SUBPROCESS_METHODS = ("sudo",)
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class CLISink(BaseSink):
    """Outputs events to console or pipes to administrative sink."""

    def __init__(self, format_type: str = "pretty", subprocess_method: str = "sudo", min_severity: str = "info"):
        self.format_type = format_type
        self.min_severity = min_severity
        if subprocess_method not in ALLOWED_SUBPROCESS_METHODS:
            raise ValueError(
                f"pipe_subprocess_method='{subprocess_method}' not allowed. "
                f"v1.0 supports only: {ALLOWED_SUBPROCESS_METHODS}."
            )
        self.subprocess_method = subprocess_method

    @property
    def name(self) -> str:
        return "cli"

    def _strip_ansi(self, text: str) -> str:
        """Strip ANSI escape sequences from formatted text."""
        return ANSI_RE.sub("", text)

    def render(self, event: ReportEvent) -> str:
        """Render event to clean plain text string (without ANSI) suitable for piping."""
        if self.format_type == "json":
            msg = json.dumps(event.to_dict())
        elif self.format_type == "raw":
            msg = f"{event.timestamp} [{event.severity.value}] {event.summary}"
        elif self.format_type == "summary":
            msg = f"[{event.severity.value.upper()}] {event.kind}: {event.summary}"
        else:  # pretty
            msg = f"[{event.severity.value.upper():8s}] [{event.kind:15s}] {event.summary}"
        return self._strip_ansi(msg)

    async def emit(self, event: ReportEvent) -> None:
        if not event.severity.is_at_least(self.min_severity):
            return
        out_stream = sys.stderr if event.severity.value in ("error", "critical") else sys.stdout
        msg = self.render(event)
        print(msg, file=out_stream, flush=True)

