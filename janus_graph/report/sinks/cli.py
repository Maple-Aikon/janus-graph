"""CLI terminal stdout report sink with secure formatting."""

from __future__ import annotations

import json
import sys
from typing import Optional
from ..models import ReportEvent
from . import BaseSink

ALLOWED_SUBPROCESS_METHODS = ("sudo",)


class CLISink(BaseSink):
    """Outputs events to console or pipes to administrative sink."""

    def __init__(self, format_type: str = "pretty", subprocess_method: str = "sudo"):
        self.format_type = format_type
        if subprocess_method not in ALLOWED_SUBPROCESS_METHODS:
            raise ValueError(
                f"pipe_subprocess_method='{subprocess_method}' not allowed. "
                f"v1.0 supports only: {ALLOWED_SUBPROCESS_METHODS}."
            )
        self.subprocess_method = subprocess_method

    @property
    def name(self) -> str:
        return "cli"

    async def emit(self, event: ReportEvent) -> None:
        out_stream = sys.stderr if event.severity.value in ("error", "critical") else sys.stdout
        if self.format_type == "json":
            msg = json.dumps(event.to_dict())
        elif self.format_type == "raw":
            msg = f"{event.timestamp} [{event.severity.value}] {event.summary}"
        else:  # pretty
            msg = f"[{event.severity.value.upper():8s}] [{event.kind:15s}] {event.summary}"
        print(msg, file=out_stream, flush=True)
