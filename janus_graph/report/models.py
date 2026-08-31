"""Data models for reporting events and notifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class ReportSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        _levels = {
            ReportSeverity.DEBUG: 10,
            ReportSeverity.INFO: 20,
            ReportSeverity.WARNING: 30,
            ReportSeverity.ERROR: 40,
            ReportSeverity.CRITICAL: 50,
        }
        return _levels.get(self, 20)

    def is_at_least(self, min_severity: Any) -> bool:
        if isinstance(min_severity, str):
            try:
                min_sev = ReportSeverity(min_severity.lower())
            except ValueError:
                return True
        elif isinstance(min_severity, ReportSeverity):
            min_sev = min_severity
        else:
            return True
        return self.level >= min_sev.level


@dataclass
class ReportEvent:
    kind: str
    severity: ReportSeverity
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "summary": self.summary,
            "details": self.details,
            "timestamp": self.timestamp,
        }
