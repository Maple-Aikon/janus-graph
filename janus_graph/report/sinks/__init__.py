"""Abstract base sink for report notifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ...core.contracts import ReportSink, Settings
from ..models import ReportEvent


class BaseSink(ABC):
    """Abstract report sink adhering to ReportSink protocol."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the sink (e.g., 'file', 'cli', 'telegram', 'webhook')."""
        pass

    @abstractmethod
    async def emit(self, event: ReportEvent) -> None:
        """Deliver a typed ReportEvent."""
        pass

    async def dispatch(self, event: Dict[str, Any], settings: Settings) -> None:
        """Protocol compliance bridge for raw event dicts."""
        # Construct ReportEvent if needed
        from ..models import ReportEvent, ReportSeverity
        severity_str = event.get("severity", "info")
        try:
            severity = ReportSeverity(severity_str)
        except ValueError:
            severity = ReportSeverity.INFO
        report_event = ReportEvent(
            kind=event.get("kind", event.get("type", "generic")),
            severity=severity,
            summary=event.get("summary", event.get("message", "")),
            details=event.get("details", event),
            timestamp=event.get("timestamp", ""),
        )
        await self.emit(report_event)
