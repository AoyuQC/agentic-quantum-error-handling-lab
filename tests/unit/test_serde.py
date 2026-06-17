"""Unit tests for the Gateway/MCP serialization helpers.

The MCP transport only preserves numerics if the round-trip is faithful: a
circuit must survive QASM serialization unchanged (modulo the terminal
measurements ``to_ir`` adds), and a calibration's numpy inverse confusion matrix
must survive the dict round-trip exactly.
"""

import numpy as np

from aqem.problems import ansatz_circuit
from aqem.tools.braket_tool import ReadoutCalibration
from aqem.tools.serde import (
    calibration_from_dict,
    calibration_to_dict,
    circuit_from_qasm,
    circuit_to_qasm,
)


def test_circuit_qasm_roundtrip_preserves_gates():
    """QASM round-trip recovers the exact gate sequence (no stray measurements)."""
    for nq in (2, 3, 4):
        circ = ansatz_circuit(nq)
        rebuilt = circuit_from_qasm(circuit_to_qasm(circ))
        orig_ops = [i.operator.name for i in circ.instructions]
        rebuilt_ops = [i.operator.name for i in rebuilt.instructions]
        assert rebuilt_ops == orig_ops
        assert "Measure" not in rebuilt_ops
        assert circ.qubit_count == rebuilt.qubit_count


def test_circuit_qasm_roundtrip_preserves_gate_angles():
    """Rotation angles survive the round-trip (so the physics is identical)."""
    circ = ansatz_circuit(2, theta1=0.37, theta2=0.21)
    rebuilt = circuit_from_qasm(circuit_to_qasm(circ))
    # Compare the OpenQASM source of the rebuilt circuit to itself round-tripped
    # again — a fixed point means the representation is stable.
    once = circuit_to_qasm(rebuilt)
    twice = circuit_to_qasm(circuit_from_qasm(once))
    assert once == twice
    assert "rx(0.37)" in once and "rx(0.21)" in once


def test_calibration_dict_roundtrip_preserves_matrix():
    """The inverse confusion matrix and errors survive the dict round-trip."""
    icm = np.array([[0.9, 0.1], [0.1, 0.9]])
    cal = ReadoutCalibration(
        inverse_confusion_matrix=icm,
        qubit_readout_errors=[0.05, 0.07],
        shots_used=12000,
    )
    rebuilt = calibration_from_dict(calibration_to_dict(cal))
    assert np.allclose(rebuilt.inverse_confusion_matrix, icm)
    assert rebuilt.qubit_readout_errors == [0.05, 0.07]
    assert rebuilt.shots_used == 12000
    # Quality is derived from the errors and must match.
    assert abs(rebuilt.quality - cal.quality) < 1e-12


def test_calibration_to_dict_is_json_safe():
    """The serialized calibration contains only JSON-native types."""
    import json

    cal = ReadoutCalibration(
        inverse_confusion_matrix=np.eye(2),
        qubit_readout_errors=[0.0, 0.0],
        shots_used=1000,
    )
    data = calibration_to_dict(cal)
    # Round-trips through json without error and is structurally identical.
    assert json.loads(json.dumps(data)) == data
