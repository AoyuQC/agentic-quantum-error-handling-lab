"""AQEM tools — the Gateway-shaped seams the DAG nodes call.

``braket_tool`` / ``vlm_tool`` hold the implementations; ``client`` is the
transport seam (in-process by default, MCP/Gateway opt-in) and ``serde`` does
the circuit/calibration JSON round-trip for the MCP boundary.
"""

from .client import (
    InProcessToolClient,
    McpToolClient,
    ToolClient,
    make_tool_client,
)

__all__ = [
    "ToolClient",
    "InProcessToolClient",
    "McpToolClient",
    "make_tool_client",
]
