"""Abstract base class for schema repair heuristic rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class HeuristicRule(ABC):
    """Abstract repair rule for handling schema discrepancies from LLM output.
    
    Adheres to the runtime_checkable RepairRule Protocol defined in core.contracts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the rule (e.g., 'edge_duplicate')."""
        pass

    @property
    @abstractmethod
    def target_model(self) -> str:
        """Primary target Pydantic class name (e.g., 'EdgeDuplicate')."""
        pass

    @property
    def target_schema_names(self) -> List[str]:
        """List of accepted schema class names (default: [target_model])."""
        return [self.target_model]

    @property
    def priority(self) -> int:
        """Execution priority. Lower values run first (default: 100)."""
        return 100

    @abstractmethod
    def can_repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> bool:
        """Check whether this rule is applicable to the given schema failure."""
        pass

    @abstractmethod
    def repair(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Dict[str, Any]:
        """Transform corrupted/incomplete payload into valid schema dictionary."""
        pass
