"""L0 acceptance: the vendored Braket+Mitiq layer runs on a local noisy
simulator, and the VLM renderer turns a plotly figure into a base64 PNG.

These exercise the third-party seams the rest of the agent builds on — no AWS.
"""

import base64

import pytest

pytestmark = pytest.mark.integration


def test_program_set_runs_on_local_noise_model():
    """A ProgramSet of simple circuits executes on qd_total and returns counts."""
    from braket.circuits import Circuit
    from braket.program_sets import ProgramSet

    from aqem.braket_mitiq.noise_models import qd_total

    # Two single-qubit prep circuits, measured.
    circuits = [Circuit().i(0).measure(0), Circuit().x(0).measure(0)]
    shots = 200
    pset = ProgramSet(circuits, shots_per_executable=shots)

    result = qd_total.run(pset, shots=shots * len(circuits)).result()

    entries = [item for entry in result for item in entry.entries]
    assert len(entries) == len(circuits)
    for item in entries:
        assert sum(item.counts.values()) == shots


def test_measurement_executor_batches_via_program_set():
    """braket_measurement_executor returns one MeasurementResult per circuit."""
    from braket.circuits import Circuit
    from mitiq import MeasurementResult

    from aqem.braket_mitiq import braket_measurement_executor
    from aqem.braket_mitiq.noise_models import qd_readout

    executor = braket_measurement_executor(qd_readout, shots=200, verbatim=False)
    circuits = [Circuit().i(0), Circuit().x(0)]

    results = executor.run(circuits)
    assert len(results) == len(circuits)
    assert all(isinstance(r, MeasurementResult) for r in results)


def test_renderer_produces_base64_png():
    """render_plot_to_base64 yields a decodable PNG (kaleido is installed)."""
    from aqem.vlm import render_plot_to_base64

    fig = {"data": [{"type": "bar", "x": ["00", "11"], "y": [0.5, 0.5]}],
           "layout": {"title": {"text": "probe"}}}

    b64 = render_plot_to_base64(fig)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number
