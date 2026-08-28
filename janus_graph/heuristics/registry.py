"""Rule registry and dispatcher for schema repairs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from .rules.base import HeuristicRule
from .rules.edge_duplicate import EdgeDuplicateRule
from .rules.extracted_edges import ExtractedEdgesRule
from .rules.extracted_entities import ExtractedEntitiesRule
from .rules.node_resolutions import NodeResolutionsRule


class HeuristicRegistry:
    """Registry managing active heuristic repair rules."""

    def __init__(self, active_rules: Optional[Sequence[str]] = None):
        self._rules: Dict[str, HeuristicRule] = {}
        self._active_rule_names: Optional[set[str]] = set(active_rules) if active_rules is not None else None
        self._register_default_rules()

    def register(self, rule: HeuristicRule) -> None:
        """Register a heuristic repair rule."""
        self._rules[rule.name] = rule

    def _register_default_rules(self) -> None:
        """Register built-in rules in order."""
        for rule_cls in (
            EdgeDuplicateRule,
            ExtractedEdgesRule,
            ExtractedEntitiesRule,
            NodeResolutionsRule,
        ):
            self.register(rule_cls())

    def get_rules(self) -> List[HeuristicRule]:
        """Return registered rules sorted by priority."""
        rules = [
            rule for name, rule in self._rules.items()
            if self._active_rule_names is None or name in self._active_rule_names
        ]
        return sorted(rules, key=lambda r: r.priority)

    def find_rule(self, schema_name: str, payload: Any, error: Optional[Exception] = None) -> Optional[HeuristicRule]:
        """Find the first matching active rule for a schema validation failure."""
        for rule in self.get_rules():
            if rule.can_repair(schema_name, payload, error):
                return rule
        return None
