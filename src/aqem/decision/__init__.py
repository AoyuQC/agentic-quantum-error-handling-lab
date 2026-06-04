"""Rules-first (VLM-second) decision logic for strategy selection and validation."""

from .rules import decide, escalate_strategy, select_strategy

__all__ = ["select_strategy", "decide", "escalate_strategy"]
