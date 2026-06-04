"""Append-only audit log of every Policy decision.

Each decision — approved or rejected — is written as one JSON line. This is the
deterministic trace that maps to AgentCore Observability later, and the record
the efficiency report draws on to show every action was budget-gated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class AuditLog:
    """In-memory audit log with optional JSONL persistence.

    Records are kept in memory (for tests and reporting) and, if ``path`` is
    given, appended to a file as JSON lines.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def approved(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("approved")]

    def rejected(self) -> list[dict[str, Any]]:
        return [r for r in self.records if not r.get("approved")]

    def __len__(self) -> int:
        return len(self.records)
