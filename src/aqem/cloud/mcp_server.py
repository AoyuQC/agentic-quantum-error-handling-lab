"""AgentCore Gateway tool server (MCP / Streamable HTTP).

Exposes the AQEM tool groups the design doc's **Gateway** routes to, as MCP
tools over a FastMCP server:

  * Tool 1 (VLM plot diagnosis): ``classify_probe``, ``validate`` — already
    JSON-in/JSON-out; the VLM client is built server-side from the environment.
  * Tool 2 (Braket + Mitiq execution): ``run_probe``, ``calibrate_readout``,
    ``run_mitigation`` — arguments arrive serialized (circuit as OpenQASM, device
    by name, calibration as a dict) and are reconstructed here, then handed to
    the exact same ``aqem.tools.braket_tool`` functions the in-process path uses,
    so the numerics are identical across transports.

Deploy as an AgentCore MCP-protocol Runtime (port 8000, ``/mcp`` endpoint):

    agentcore configure -e agent_mcp.py -n aqem-tools --protocol MCP
    agentcore deploy

Run locally for tests / development:

    python -m aqem.cloud.mcp_server          # serves http://0.0.0.0:8000/mcp
"""

from __future__ import annotations

import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..config import resolve_device
from ..models import Strategy
from ..tools import braket_tool, vlm_tool
from ..tools.serde import calibration_from_dict, circuit_from_qasm

mcp = FastMCP(
    host="0.0.0.0",
    port=int(os.environ.get("AQEM_MCP_PORT", "8000")),
    stateless_http=True,
)


@mcp.custom_route("/ping", methods=["GET"])
async def _ping(request):  # noqa: ANN001
    """Health check the AgentCore MCP Runtime polls on port 8000.

    FastMCP only serves ``/mcp`` by default; AgentCore's container contract also
    requires ``GET /ping`` to return 200, so add it explicitly.
    """
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "healthy"})


def _build_vlm():
    """Build the VLM client from the environment (server-side).

    Returns ``None`` if construction fails so the VLM tools degrade to a rules
    fallback (matching ``vlm_tool``'s contract) rather than erroring the call.
    """
    from ..vlm import get_vlm_client

    try:
        return get_vlm_client(
            {
                "provider": os.environ.get("AQEM_VLM_PROVIDER", "bedrock"),
                "model_id": os.environ.get(
                    "AQEM_VLM_MODEL_ID", "us.anthropic.claude-opus-4-8"
                ),
                "endpoint": os.environ.get("AQEM_VLM_ENDPOINT", ""),
                "region": os.environ.get("AWS_REGION", "us-east-1"),
                "temperature": 0,
                "max_tokens": 4096,
            }
        )
    except Exception:  # missing config / unknown provider -> degrade, don't crash
        return None


# --- Tool 1: VLM plot diagnosis -------------------------------------------
@mcp.tool()
def classify_probe(
    plots: list[dict[str, Any]], confidence_threshold: float = 0.5
) -> dict[str, Any]:
    """Classify the dominant error source from probe histograms (VLM)."""
    return vlm_tool.classify_probe_with_vlm(_build_vlm(), plots, confidence_threshold)


@mcp.tool()
def validate(
    plots: list[dict[str, Any]], confidence_threshold: float = 0.5
) -> dict[str, Any]:
    """Judge a ZNE extrapolation plot and recommend stop / retry mode (VLM)."""
    return vlm_tool.validate_with_vlm(_build_vlm(), plots, confidence_threshold)


# --- Tool 2: Braket + Mitiq execution -------------------------------------
@mcp.tool()
def run_probe(circuit_qasm: str, device_name: str, shots: int) -> dict[str, int]:
    """Run one probe circuit on the named noise model; return raw counts."""
    circuit = circuit_from_qasm(circuit_qasm)
    device = resolve_device(device_name)
    counts = braket_tool.run_probe(circuit, device, int(shots))
    return {str(k): int(v) for k, v in counts.items()}


@mcp.tool()
def calibrate_readout(
    num_qubits: int, device_name: str, rem_twirls: int, shots: int
) -> dict[str, Any]:
    """Build the REM inverse confusion matrix; return its serializable summary."""
    from ..tools.serde import calibration_to_dict

    device = resolve_device(device_name)
    cal = braket_tool.calibrate_readout(
        int(num_qubits), device, int(rem_twirls), int(shots)
    )
    return calibration_to_dict(cal)


@mcp.tool()
def run_mitigation(
    circuit_qasm: str,
    observable: list[list[Any]],
    device_name: str,
    strategy: dict[str, Any],
    calibration: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the chosen {REM, PT, ZNE} subset; return the mitigated estimate."""
    circuit = circuit_from_qasm(circuit_qasm)
    device = resolve_device(device_name)
    obs = [(float(c), p) for c, p in observable]
    strat = Strategy.from_dict(strategy)
    cal = calibration_from_dict(calibration) if calibration else None
    result = braket_tool.run_mitigation(circuit, obs, device, strat, cal)
    return {
        "value": result.value,
        "error_bar": result.error_bar,
        "shots_used": result.shots_used,
        "zne_data": result.zne_data,
        "techniques": result.techniques,
    }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
