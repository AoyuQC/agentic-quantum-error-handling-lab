"""Bedrock Guardrails wrapper, alongside the deterministic ``policy/`` layer.

The deterministic Policy (controlled action set + budget gate + no-recalibration
guard) remains the authoritative safety mechanism. Bedrock Guardrails add a
managed content-safety check on free-text I/O (the task prompt in, the report
text out). When no guardrail is configured the wrapper is a transparent no-op,
so local dev and tests are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    allowed: bool
    action: str          # "NONE" | "GUARDRAIL_INTERVENED" | "DISABLED" | "ERROR"
    reason: str = ""


class Guardrail:
    """Optional Bedrock Guardrails check around agent text I/O.

    Args:
        guardrail_id: Bedrock guardrail identifier; None disables the check.
        guardrail_version: guardrail version (default "DRAFT").
        region: AWS region for the bedrock-runtime client.
        fail_open: if a guardrail call errors, allow (True) or block (False).
    """

    def __init__(
        self,
        guardrail_id: Optional[str] = None,
        guardrail_version: str = "DRAFT",
        region: Optional[str] = None,
        fail_open: bool = True,
    ):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.region = region
        self.fail_open = fail_open
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.guardrail_id)

    def _runtime(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def check(self, text: str, source: str = "INPUT") -> GuardrailResult:
        """Apply the guardrail to ``text``. ``source`` is "INPUT" or "OUTPUT"."""
        if not self.enabled:
            return GuardrailResult(allowed=True, action="DISABLED")
        try:
            resp = self._runtime().apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=[{"text": {"text": text}}],
            )
            action = resp.get("action", "NONE")
            allowed = action != "GUARDRAIL_INTERVENED"
            reason = "" if allowed else "guardrail intervened on content"
            return GuardrailResult(allowed=allowed, action=action, reason=reason)
        except Exception as e:  # network / perms / unknown guardrail
            return GuardrailResult(
                allowed=self.fail_open, action="ERROR", reason=str(e)
            )
