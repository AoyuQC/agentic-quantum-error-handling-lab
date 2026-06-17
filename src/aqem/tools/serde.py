"""Serialization helpers for the Gateway/MCP tool boundary.

The in-process tools pass live objects — a Braket ``Circuit``, a ``Device``, and
a numpy-backed ``ReadoutCalibration`` — directly between DAG nodes. To route the
same calls through an MCP server those arguments must become JSON. These helpers
do the round-trip and are written so that the server-side reconstruction yields
numerically identical results (see ``tests/unit/test_serde.py``):

  * circuit  <-> OpenQASM 3.0 source (terminal measurements stripped on the way
    back, since the mitigation runner adds its own measurement bases);
  * ReadoutCalibration <-> the serializable ``models.Calibration`` summary, whose
    inverse confusion matrix is a nested list.

Devices are passed by *name* (e.g. ``"qd_readout_2"``) and re-resolved on the
server via ``config.resolve_device`` — the noise model is code, not data.
"""

from __future__ import annotations

import numpy as np
from braket.circuits import Circuit

from ..models import Calibration
from .braket_tool import ReadoutCalibration


def circuit_to_qasm(circuit: Circuit) -> str:
    """Serialize a Braket circuit to OpenQASM 3.0 source."""
    return circuit.to_ir(ir_type="OPENQASM").source


def circuit_from_qasm(qasm: str) -> Circuit:
    """Rebuild a measurement-free circuit from OpenQASM 3.0 source.

    ``Circuit.from_ir`` re-adds the terminal ``measure`` instructions that
    ``to_ir`` emitted; the AQEM target/probe circuits are defined without them
    (the mitigation runner and probes add their own measurements), so strip any
    trailing ``Measure`` operators to recover the original circuit exactly.
    """
    parsed = Circuit.from_ir(qasm)
    rebuilt = Circuit()
    for instr in parsed.instructions:
        if instr.operator.name == "Measure":
            continue
        rebuilt.add_instruction(instr)
    return rebuilt


def calibration_to_dict(cal: ReadoutCalibration) -> dict:
    """Serialize a live ReadoutCalibration to a JSON-safe dict.

    Reuses the :class:`~aqem.models.Calibration` summary shape so the audit
    log, the node output, and the MCP payload all share one representation.
    """
    return Calibration(
        inverse_confusion_matrix=np.asarray(cal.inverse_confusion_matrix).tolist(),
        qubit_readout_errors=[float(e) for e in cal.qubit_readout_errors],
        quality=cal.quality,
        shots_used=cal.shots_used,
    ).to_dict()


def calibration_from_dict(data: dict) -> ReadoutCalibration:
    """Rebuild a live ReadoutCalibration (numpy ICM) from its dict form."""
    return ReadoutCalibration(
        inverse_confusion_matrix=np.asarray(
            data["inverse_confusion_matrix"], dtype=float
        ),
        qubit_readout_errors=[float(e) for e in data.get("qubit_readout_errors", [])],
        shots_used=int(data.get("shots_used", 0)),
    )
