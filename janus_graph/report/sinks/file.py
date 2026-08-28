"""JSONL file report sink with size-based rotation support."""

from __future__ import annotations

import json
import os
from pathlib import Path
from ..models import ReportEvent
from . import BaseSink


class FileSink(BaseSink):
    """Writes report events to a rotating JSONL file."""

    def __init__(self, path: str = "./data/logs/janus_report.jsonl", rotation_max_bytes: int = 10485760):
        self.path = Path(path).resolve()
        self.rotation_max_bytes = rotation_max_bytes

    @property
    def name(self) -> str:
        return "file"

    def _rotate_if_needed(self) -> None:
        if self.path.exists() and self.path.stat().st_size >= self.rotation_max_bytes:
            rotated = self.path.with_name(f"{self.path.name}.1")
            try:
                if rotated.exists():
                    rotated.unlink()
                self.path.rename(rotated)
            except Exception:
                pass

    async def emit(self, event: ReportEvent) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            record = event.to_dict()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass
