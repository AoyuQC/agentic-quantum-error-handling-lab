"""AgentCore Gateway MCP-server entrypoint (target for `agentcore configure`).

Re-exports the FastMCP ``mcp`` app built in ``aqem.cloud.mcp_server`` — the
Gateway tool server exposing the Braket/Mitiq + VLM tools over MCP. Deploy as an
MCP-protocol Runtime (port 8000, ``/mcp`` endpoint):

    agentcore configure --entrypoint agent_mcp.py --name aqem-tools --protocol MCP
    agentcore deploy --agent aqem-tools
"""

from aqem.cloud.mcp_server import mcp  # noqa: F401

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
