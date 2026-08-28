"""Abstract base class for cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheBackend(ABC):
    """Abstract cache storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def size(self) -> int:
        pass
