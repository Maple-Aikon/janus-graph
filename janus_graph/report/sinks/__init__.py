"""Sinks package for report event destinations."""

from abc import ABC, abstractmethod
from ..models import ReportEvent


class BaseSink(ABC):
    """Abstract reporting destination sink."""

    @abstractmethod
    async def emit(self, event: ReportEvent) -> None:
        pass
