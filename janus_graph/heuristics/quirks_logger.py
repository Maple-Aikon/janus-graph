"""JSONL Audit logger for schema repair events and model quirks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class QuirksLogger:
    """Logs schema validation failures and repair actions for audit.
    
    Implements QuirksLogger Protocol from core.contracts.
    """

    def __init__(self, log_path: str = "./data/logs/llm_schema_quirks.jsonl"):
        self.log_path = Path(log_path).resolve()

    def log_schema_quirk(
        self,
        *,
        model_name: str,
        field: str,
        observed_type: type,
        expected_type: type,
        repair_applied: bool,
        episode_id: str,
    ) -> None:
        """Protocol method: log a specific schema discrepancy."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "schema_quirk",
                "episode_id": episode_id,
                "model_name": model_name,
                "field": field,
                "observed_type": str(observed_type),
                "expected_type": str(expected_type),
                "repair_applied": repair_applied,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def log_repair(
        self,
        rule_name: str,
        schema_name: str,
        original_payload: Any,
        repaired_payload: Dict[str, Any],
        error_message: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """Append a repair event to the quirks JSONL log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "repair",
                "episode_id": episode_id or "unknown",
                "rule": rule_name,
                "schema": schema_name,
                "error": error_message,
                "original": original_payload,
                "repaired": repaired_payload,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass
