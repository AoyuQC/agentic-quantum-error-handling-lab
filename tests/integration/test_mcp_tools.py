"""Integration test: the AgentCore Gateway (MCP) tool path end-to-end.

Boots the FastMCP tool server (``aqem.cloud.mcp_server``) in a subprocess and
drives it with a real ``McpToolClient`` over Streamable HTTP — the same client
the loop uses when ``AQEM_TOOL_TRANSPORT=mcp``. Asserts the Braket/Mitiq tools
work over the wire and that an MCP-run mitigation matches an in-process run
closely (the simulator's shot sampling is not seedable, so we assert closeness,
not bitwise equality — consistent with the rest of the integration suite).

No AWS: the noise model is a local simulator and the VLM is not exercised here
(server-side VLM needs Bedrock creds; the VLM path is covered offline in
``tests/unit/test_tool_client.py`` + ``test_vlm_tool.py``).
"""

import os
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_endpoint():
    """Spawn the MCP tool server on a free port; yield its /mcp endpoint."""
    port = _free_port()
    # Force the server-side VLM to a provider it can't reach so the VLM tools
    # degrade deterministically (no live Bedrock call, no ambient-creds flakiness).
    env = {
        **os.environ,
        "AQEM_MCP_PORT": str(port),
        "AQEM_VLM_PROVIDER": "custom",
        "AQEM_VLM_ENDPOINT": "http://127.0.0.1:1/unreachable",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "aqem.cloud.mcp_server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    endpoint = f"http://127.0.0.1:{port}/mcp"
    # Wait for the port to accept connections.
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"MCP server exited early:\n{out}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                ready = True
                break
        except OSError:
            time.sleep(0.3)
    if not ready:
        proc.terminate()
        raise RuntimeError("MCP server did not start in time")
    yield endpoint
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _client(mcp_endpoint):
    from aqem.tools.client import McpToolClient

    return McpToolClient(
        endpoint=mcp_endpoint, device_name="qd_readout_2", vlm_enabled=False
    )


def test_run_probe_over_mcp_returns_counts(mcp_endpoint):
    """run_probe across the wire returns a normalized counts histogram."""
    from aqem.probes.circuits import probe_circuits

    client = _client(mcp_endpoint)
    readout = probe_circuits(2)["readout"]
    counts = client.run_probe(readout, shots=2000)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 2000
    # Readout probe prepares |00>; most mass sits on the all-zeros bar.
    assert counts.get("00", 0) > sum(counts.values()) * 0.5


def test_calibrate_readout_over_mcp_rebuilds_matrix(mcp_endpoint):
    """calibrate_readout across the wire yields a usable ReadoutCalibration."""
    import numpy as np

    client = _client(mcp_endpoint)
    cal = client.calibrate_readout(num_qubits=2, rem_twirls=20, shots=4000)
    icm = np.asarray(cal.inverse_confusion_matrix)
    assert icm.shape == (4, 4)
    assert 0.0 <= cal.quality <= 1.0
    assert cal.shots_used == 4000


def test_run_mitigation_over_mcp_matches_inprocess(mcp_endpoint):
    """An MCP-routed mitigation matches an in-process run within shot noise."""
    from aqem.braket_mitiq.noise_models import qd_readout_2
    from aqem.models import Strategy
    from aqem.problems import default_problem, ideal_expectation
    from aqem.tools.client import InProcessToolClient

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    ideal = ideal_expectation(circuit, problem.observable)
    strategy = Strategy(techniques=["REM"], twirl_count=8, rem_twirls=20)

    # In-process reference.
    inproc = InProcessToolClient(device=qd_readout_2, vlm=None)
    cal_ip = inproc.calibrate_readout(2, strategy.rem_twirls, strategy.shot_per_base * strategy.overhead)
    res_ip = inproc.run_mitigation(circuit, problem.observable, strategy, cal_ip)

    # Same call routed over MCP (server rebuilds circuit/device/calibration).
    client = _client(mcp_endpoint)
    cal_mcp = client.calibrate_readout(2, strategy.rem_twirls, strategy.shot_per_base * strategy.overhead)
    res_mcp = client.run_mitigation(circuit, problem.observable, strategy, cal_mcp)

    assert "REM" in res_mcp.techniques == res_ip.techniques
    assert res_mcp.shots_used == res_ip.shots_used
    # Both estimate the same observable with the same recipe; each is near ideal
    # (REM on a readout-dominated device). The simulator's shot sampling is not
    # seedable, so allow a few error-bars of slack rather than bitwise equality
    # — the point is that the MCP path produces the same physics, not the same
    # random draw.
    assert abs(res_mcp.value - ideal) < 0.3
    assert abs(res_mcp.value - res_ip.value) < 0.3


def test_adaptive_loop_runs_over_mcp_transport(mcp_endpoint):
    """The whole adaptive loop runs with every tool call routed through MCP."""
    from aqem.braket_mitiq.noise_models import qd_readout_2
    from aqem.loop import run_adaptive_loop
    from aqem.models import Budget
    from aqem.problems import default_problem
    from aqem.tools.client import McpToolClient

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    tools = McpToolClient(
        endpoint=mcp_endpoint, device_name="qd_readout_2", vlm_enabled=False
    )
    record = run_adaptive_loop(
        problem,
        circuit,
        qd_readout_2,
        Budget(shots_total=2_000_000),
        config={
            "probe_shots": 2000,
            "shot_per_base": 4000,
            "overhead": 3,
            "rem_twirls": 20,
            "use_ideal_for_validation": True,
        },
        tools=tools,
        seed=7,
    )
    # Same behavioural invariants as the in-process loop test.
    assert record.status in ("stopped", "exhausted")
    est = record.final_outputs["estimate"]
    assert est is not None
    assert "REM" in est["techniques"]


def test_classify_probe_over_mcp_degrades_without_vlm(mcp_endpoint, monkeypatch):
    """With VLM disabled server-side, classify_probe returns a degraded verdict.

    The server builds its VLM from the environment; pointing it at a provider it
    can't reach (no creds) should surface as a graceful ``degraded`` result, not
    a transport error — proving the VLM tool is reachable over MCP.
    """
    from aqem.tools.client import McpToolClient

    client = McpToolClient(
        endpoint=mcp_endpoint, device_name="qd_readout_2", vlm_enabled=True
    )
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    plots = [{"name": "p", "format": "png", "data": tiny_png}]
    out = client.classify_probe(plots, confidence_threshold=0.5)
    assert isinstance(out, dict)
    assert "degraded" in out
