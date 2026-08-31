"""SchemaRepairingLLMClient wrapping Graphiti LLMClient with self-repair.

Compatibility shim for graphiti_core >=0.20:
  - The upstream API surface changed: ``_send`` / ``_send_json`` / ``_send_response_model``
    no longer exist on ``LLMClient``. The public entrypoint is now
    ``generate_response(messages, response_model=None, ...) -> dict`` which returns a
    ``dict`` (validation against ``response_model`` happens at the Graphiti call site,
    not inside the client).
  - ``_generate_response`` is the abstract hook ``OpenAIGenericClient`` (and any custom
    subclass) implements; we override it here as a thin delegate so the wrapper remains
    instantiable.
  - Schema repair intercepts the ``ValidationError`` raised at the Graphiti-level
    call site (``response_model(**response)``) by overriding ``generate_response``
    and validating the returned ``dict`` against the requested ``response_model``
    before returning. If validation fails, we run the heuristic registry to repair
    the payload in-place, log the quirk, and return the repaired dict. Otherwise we
    delegate unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel, ValidationError

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
        # graphiti_core LLMClient.__init__ expects (config, cache). We delegate all
        # config/token plumbing to ``inner``, so pass inner's config through and
        # disable caching on the wrapper layer.
        inner_config = getattr(inner, "config", None)
        super().__init__(inner_config, cache=False)
        self.inner = inner
        self.registry = HeuristicRegistry(active_rules=active_rules)
        self.quirks_logger = QuirksLogger(log_path=quirks_log_path)
        # Mirror token/temperature attributes from inner so Graphiti doesn't
        # observe an empty client after construction.
        for attr in ("model", "small_model", "temperature", "max_tokens"):
            setattr(self, attr, getattr(inner, attr, getattr(self, attr, None)))

    # ------------------------------------------------------------------ #
    # Abstract hook (graphiti_core >=0.20): delegate to inner.
    # ------------------------------------------------------------------ #
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> Dict[str, Any]:
        return await self.inner._generate_response(
            messages, response_model, max_tokens, model_size
        )

    # ------------------------------------------------------------------ #
    # Public entrypoint override: repair payloads that would otherwise raise
    # ``ValidationError`` at the Graphiti call site (``response_model(**payload)``).
    # ------------------------------------------------------------------ #
    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> Dict[str, Any]:
        response = await self.inner.generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
        )

        if response_model is None or not isinstance(response, dict):
            return response

        # Try normal validation; on failure, run heuristic repair.
        try:
            response_model.model_validate(response)
            return response
        except ValidationError as err:
            schema_name = getattr(response_model, "__name__", str(response_model))
            logger.debug("Schema validation error caught for %s: %s", schema_name, err)

            # Prefer the offending payload from err.errors()[0]["input"] when present
            # (the same convention the heuristics pipeline relies on), falling back
            # to the full response dict.
            raw_input: Any = response
            if err.errors():
                raw_input = err.errors()[0].get("input", response)

            rule = self.registry.find_rule(schema_name, raw_input, err)
            if rule is None:
                # No rule matches — surface original ValidationError to caller.
                raise

            repaired = rule.repair(schema_name, raw_input, err)
            self.quirks_logger.log_repair(
                rule_name=rule.name,
                schema_name=schema_name,
                original_payload=raw_input,
                repaired_payload=repaired,
                error_message=str(err),
            )
            # Re-validate the repaired payload to confirm it actually parses.
            response_model.model_validate(repaired)
            return repaired
