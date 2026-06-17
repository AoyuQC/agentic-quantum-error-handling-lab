"""Cloud wrap (Phase C5): AgentCore Runtime entrypoint, S3 artifacts, Guardrails.

The VLM remains managed Claude on Bedrock. Tools run in-process by default; an
opt-in **Gateway** MCP server (``mcp_server``, deployed via ``agent_mcp.py``)
exposes the same ``aqem.tools`` functions over the wire when
``AQEM_TOOL_TRANSPORT=mcp``. Everything here degrades to local behaviour when AWS
resources are not configured, so the same code runs in tests and under AgentCore.
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
