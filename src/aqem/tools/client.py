"""Tool transport seam — in-process (default) or MCP/Gateway.

The DAG nodes call their tools through ``ctx.tools`` rather than importing the
``braket_tool`` / ``vlm_tool`` functions directly. This indirection is the
"Gateway" the design doc describes: by default the calls run **in-process**
(``InProcessToolClient`` — identical numerics, offline, no AWS), and when opted
in via ``AQEM_TOOL_TRANSPORT=mcp`` they route over the wire to a FastMCP server
(``McpToolClient``) such as the live AgentCore MCP Runtime.

The client holds the two per-run constants — the Braket ``device`` and the VLM
client — so node call sites pass only what varies (circuit, plots, params) and
are oblivious to the transport. The factory mirrors
``cloud.artifacts.make_artifact_store``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol, runtime_checkable

from .braket_tool import (
    MitigationResult,
    ReadoutCalibration,
    calibrate_readout,
    run_mitigation,
    run_probe,
)
from .vlm_tool import classify_probe_with_vlm, validate_with_vlm


@runtime_checkable
class ToolClient(Protocol):
    """The named tool API the DAG nodes call (device + VLM are bound in)."""

    @property
    def vlm_enabled(self) -> bool: ...

    def run_probe(self, circuit: Any, shots: int) -> dict[str, int]: ...

    def calibrate_readout(
        self, num_qubits: int, rem_twirls: int, shots: int
    ) -> ReadoutCalibration: ...

    def run_mitigation(
        self,
        circuit: Any,
        observable: list[tuple[float, str]],
        strategy: Any,
        calibration: Optional[ReadoutCalibration],
    ) -> MitigationResult: ...

    def classify_probe(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]: ...

    def validate(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]: ...


class InProcessToolClient:
    """Default transport: call the tool functions directly (no serialization)."""

    def __init__(self, device: Any, vlm: Any = None, device_name: Optional[str] = None):
        self._device = device
        self._vlm = vlm
        self._device_name = device_name

    @property
    def vlm_enabled(self) -> bool:
        return self._vlm is not None

    def run_probe(self, circuit: Any, shots: int) -> dict[str, int]:
        return run_probe(circuit, self._device, shots)

    def calibrate_readout(
        self, num_qubits: int, rem_twirls: int, shots: int
    ) -> ReadoutCalibration:
        return calibrate_readout(num_qubits, self._device, rem_twirls, shots)

    def run_mitigation(
        self,
        circuit: Any,
        observable: list[tuple[float, str]],
        strategy: Any,
        calibration: Optional[ReadoutCalibration],
    ) -> MitigationResult:
        return run_mitigation(circuit, observable, self._device, strategy, calibration)

    def classify_probe(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]:
        return classify_probe_with_vlm(self._vlm, plots, confidence_threshold)

    def validate(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]:
        return validate_with_vlm(self._vlm, plots, confidence_threshold)


class McpToolClient:
    """MCP transport: route every tool call to a FastMCP server (the Gateway).

    The braket arguments are serialized (circuit -> QASM, device -> name,
    calibration -> dict) and the server reconstructs them and calls the very
    same ``braket_tool`` / ``vlm_tool`` functions, so results are numerically
    identical to the in-process path. The VLM runs server-side; ``vlm_enabled``
    reflects whether the caller asked for VLM steering.
    """

    def __init__(
        self,
        endpoint: str,
        device_name: str,
        vlm_enabled: bool = True,
        headers: Optional[dict[str, str]] = None,
        sigv4: Optional[bool] = None,
        region: Optional[str] = None,
    ):
        self._endpoint = endpoint
        self._device_name = device_name
        self._vlm_enabled = bool(vlm_enabled)
        self._headers = headers or {}
        # A live AgentCore-hosted MCP runtime is invoked at the bedrock-agentcore
        # data plane and must be SigV4-signed; a plain local server is not.
        # Auto-detect from the endpoint host unless told explicitly.
        self._sigv4 = (
            sigv4 if sigv4 is not None else "bedrock-agentcore" in endpoint
        )
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")

    @property
    def vlm_enabled(self) -> bool:
        return self._vlm_enabled

    # -- transport ----------------------------------------------------------
    def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke one MCP tool over Streamable HTTP and return its result."""
        from .vlm_tool import _run_async  # reuse the sync<->async bridge

        return _run_async(self._call_async(tool, arguments))

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        auth = _SigV4Auth(self._region) if self._sigv4 else None
        async with streamablehttp_client(
            self._endpoint, headers=self._headers, auth=auth
        ) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                if getattr(result, "isError", False):
                    text = _result_text(result)
                    raise RuntimeError(f"MCP tool {tool} failed: {text}")
                return _result_payload(result)

    # -- tool API -----------------------------------------------------------
    def run_probe(self, circuit: Any, shots: int) -> dict[str, int]:
        from .serde import circuit_to_qasm

        counts = self._call(
            "run_probe",
            {
                "circuit_qasm": circuit_to_qasm(circuit),
                "device_name": self._device_name,
                "shots": int(shots),
            },
        )
        return {str(k): int(v) for k, v in counts.items()}

    def calibrate_readout(
        self, num_qubits: int, rem_twirls: int, shots: int
    ) -> ReadoutCalibration:
        from .serde import calibration_from_dict

        data = self._call(
            "calibrate_readout",
            {
                "num_qubits": int(num_qubits),
                "device_name": self._device_name,
                "rem_twirls": int(rem_twirls),
                "shots": int(shots),
            },
        )
        return calibration_from_dict(data)

    def run_mitigation(
        self,
        circuit: Any,
        observable: list[tuple[float, str]],
        strategy: Any,
        calibration: Optional[ReadoutCalibration],
    ) -> MitigationResult:
        from .serde import calibration_to_dict, circuit_to_qasm

        data = self._call(
            "run_mitigation",
            {
                "circuit_qasm": circuit_to_qasm(circuit),
                "observable": [[float(c), p] for c, p in observable],
                "device_name": self._device_name,
                "strategy": strategy.to_dict(),
                "calibration": calibration_to_dict(calibration) if calibration else None,
            },
        )
        return MitigationResult(
            value=float(data["value"]),
            error_bar=float(data["error_bar"]),
            shots_used=int(data["shots_used"]),
            zne_data={str(k): float(v) for k, v in data.get("zne_data", {}).items()},
            techniques=list(data.get("techniques", [])),
        )

    def classify_probe(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]:
        return self._call(
            "classify_probe",
            {"plots": plots, "confidence_threshold": float(confidence_threshold)},
        )

    def validate(
        self, plots: list[dict[str, Any]], confidence_threshold: float
    ) -> dict[str, Any]:
        return self._call(
            "validate",
            {"plots": plots, "confidence_threshold": float(confidence_threshold)},
        )


def _sigv4_auth_cls():
    """Build the httpx SigV4 auth class lazily (botocore is a cloud-only dep)."""
    import httpx
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.session import Session

    class _Auth(httpx.Auth):
        """Sign each request for the ``bedrock-agentcore`` service with SigV4."""

        requires_request_body = True

        def __init__(self, region: str):
            self._region = region
            self._credentials = Session().get_credentials()

        def auth_flow(self, request):  # noqa: ANN001
            aws_req = AWSRequest(
                method=request.method,
                url=str(request.url),
                data=request.content,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    # botocore re-adds these; signing the host-derived ones is fine.
                    if k.lower() not in ("connection",)
                },
            )
            SigV4Auth(self._credentials, "bedrock-agentcore", self._region).add_auth(
                aws_req
            )
            request.headers.update(dict(aws_req.headers))
            yield request

    return _Auth


def _SigV4Auth(region: str):  # noqa: N802  (factory returning an instance)
    return _sigv4_auth_cls()(region)


def _result_payload(result: Any) -> Any:
    """Extract the tool's return value from an MCP CallToolResult."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        # FastMCP wraps non-dict returns as {"result": value}.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    text = _result_text(result)
    return json.loads(text) if text else None


def _result_text(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def make_tool_client(
    *,
    device: Any = None,
    vlm: Any = None,
    device_name: Optional[str] = None,
    transport: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> ToolClient:
    """Build the tool client for the configured transport.

    ``transport`` (or ``AQEM_TOOL_TRANSPORT``) is ``"inprocess"`` (default) or
    ``"mcp"``. The MCP path needs ``device_name`` and an ``endpoint`` (or
    ``AQEM_MCP_ENDPOINT``); it falls back to in-process if those are missing.
    """
    transport = (transport or os.environ.get("AQEM_TOOL_TRANSPORT", "inprocess")).lower()
    if transport == "mcp":
        endpoint = endpoint or os.environ.get("AQEM_MCP_ENDPOINT")
        if endpoint and device_name:
            return McpToolClient(
                endpoint=endpoint,
                device_name=device_name,
                vlm_enabled=vlm is not None,
            )
        # Misconfigured MCP transport — degrade to in-process rather than fail.
    return InProcessToolClient(device=device, vlm=vlm, device_name=device_name)
