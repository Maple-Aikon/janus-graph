"""SchemaRepairingLLMClient wrapping Graphiti LLMClient with self-repair."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from graphiti_core.llm_client import LLMClient
from pydantic import ValidationError
from .quirks_logger import QuirksLogger
from .registry import HeuristicRegistry

logger = logging.getLogger("janus_graph.heuristics.repairing_client")


class SchemaRepairingLLMClient(LLMClient):
    """LLM client wrapper that intercepts Pydantic ValidationErrors and self-repairs.
    
    Inherits from Graphiti's LLMClient ABC to pass isinstance validation checks.
    """

    def __init__(
        self,
        inner: LLMClient,
        active_rules: Optional[Sequence[str]] = None,
        quirks_log_path: str = "./data/logs/llm_schema_quirks.jsonl",
    ):
        self.inner = inner
        self.registry = HeuristicRegistry(active_rules=active_rules)
        self.quirks_logger = QuirksLogger(log_path=quirks_log_path)

    async def _send(self, messages: list[dict[str, str]]) -> str:
        return await self.inner._send(messages)

    async def _send_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        return await self.inner._send_json(messages, schema)

    async def _send_response_model(
        self,
        messages: list[dict[str, str]],
        response_model: type[Any],
    ) -> Any:
        try:
            return await self.inner._send_response_model(messages, response_model)
        except ValidationError as err:
            schema_name = getattr(response_model, "__name__", str(response_model))
            logger.warning("Schema validation error caught for %s: %s", schema_name, err)

            raw_input: Any = {}
            if err.errors():
                raw_input = err.errors()[0].get("input", {})

            rule = self.registry.find_rule(schema_name, raw_input, err)
            if rule:
                repaired = rule.repair(schema_name, raw_input, err)
                self.quirks_logger.log_repair(
                    rule_name=rule.name,
                    schema_name=schema_name,
                    original_payload=raw_input,
                    repaired_payload=repaired,
                    error_message=str(err),
                )
                if hasattr(response_model, "model_validate"):
                    return response_model.model_validate(repaired)
                return response_model(**repaired)
            raise
