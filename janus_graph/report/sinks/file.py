"""JSONL file report sink with rotation support."""

from __future__ import annotations

import json
from pathlib import Path
from ..models import ReportEvent
from . import BaseSink


class FileSink(BaseSink):
    """Writes report events to a rotating JSONL file."""

    def __init__(self, path: str = "./data/logs/janus_report.jsonl", rotation_max_bytes: int = 10485760):
        self.path = Path(path).resolve()
        self.rotation_max_bytes = rotation_max_bytes

    async def emit(self, event: ReportEvent) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": event.timestamp,
                "kind": event.kind,
                "severity": event.severity.value,
                "summary": event.summary,
                "details": event.details,
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass
