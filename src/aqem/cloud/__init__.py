"""Cloud wrap (Phase C5): AgentCore Runtime entrypoint, S3 artifacts, Guardrails.

The VLM remains managed Claude on Bedrock. Tools run in-process (the seams in
``aqem.tools`` are where a Gateway MCP server would later route). Everything here
degrades to local behaviour when AWS resources are not configured, so the same
code runs in tests and under AgentCore Runtime.
"""

from .artifacts import ArtifactStore, make_artifact_store
from .guardrails import Guardrail, GuardrailResult
from .runtime import handle, invoke

__all__ = [
    "ArtifactStore",
    "make_artifact_store",
    "Guardrail",
    "GuardrailResult",
    "handle",
    "invoke",
]
