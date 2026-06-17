"""Unit tests for the tool transport seam (``aqem.tools.client``).

Covers the factory's transport routing and the in-process client's behaviour —
the default path the whole offline suite relies on. The live MCP path is
exercised in ``tests/integration/test_mcp_tools.py``.
"""

import sys
from pathlib import Path

from aqem.tools.client import (
    InProcessToolClient,
    McpToolClient,
    make_tool_client,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.fake_vlm import FakeVLM  # noqa: E402


def test_factory_defaults_to_inprocess():
    client = make_tool_client(device="DEV", vlm=None, device_name="qd_readout_2")
    assert isinstance(client, InProcessToolClient)


def test_factory_inprocess_explicit():
    client = make_tool_client(
        device="DEV", vlm=None, device_name="qd_readout_2", transport="inprocess"
    )
    assert isinstance(client, InProcessToolClient)


def test_factory_mcp_with_endpoint():
    client = make_tool_client(
        device=None,
        vlm=FakeVLM("{}"),
        device_name="qd_readout_2",
        transport="mcp",
        endpoint="http://localhost:8000/mcp",
    )
    assert isinstance(client, McpToolClient)
    assert client.vlm_enabled is True


def test_factory_mcp_missing_endpoint_degrades_to_inprocess():
    """A misconfigured MCP transport falls back rather than crashing the loop."""
    client = make_tool_client(
        device="DEV", vlm=None, device_name="qd_readout_2", transport="mcp", endpoint=None
    )
    assert isinstance(client, InProcessToolClient)


def test_factory_reads_env(monkeypatch):
    monkeypatch.setenv("AQEM_TOOL_TRANSPORT", "mcp")
    monkeypatch.setenv("AQEM_MCP_ENDPOINT", "http://localhost:8000/mcp")
    client = make_tool_client(device=None, vlm=None, device_name="qd_readout_2")
    assert isinstance(client, McpToolClient)


def test_inprocess_vlm_enabled_reflects_client():
    assert InProcessToolClient(device=None, vlm=None).vlm_enabled is False
    assert InProcessToolClient(device=None, vlm=FakeVLM("{}")).vlm_enabled is True


def test_inprocess_classify_probe_delegates_to_vlm_tool():
    """The in-process client forwards plots to the VLM and returns its verdict."""
    vlm = FakeVLM(
        {
            "dominant_error": "readout",
            "suggested_focus": ["REM"],
            "confidence": 0.9,
            "rationale": "mass off the all-zeros bar",
        }
    )
    client = InProcessToolClient(device=None, vlm=vlm)
    # A pre-rendered base64 PNG plot so no kaleido/plotly is needed here.
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    plots = [{"name": "p", "format": "png", "data": tiny_png}]
    out = client.classify_probe(plots, confidence_threshold=0.5)
    assert out["degraded"] is False
    assert out["dominant_error"] == "readout"
    assert vlm.calls and vlm.calls[0]["n_images"] == 1
