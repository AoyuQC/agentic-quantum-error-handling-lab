"""Cheap diagnostic probe circuits.

These run a handful of shots to characterize noise *empirically* — no device
metadata (design-doc §1.3 / §2.2). Two probes:

  - readout calibration: prepare a known computational-basis state (all-zeros)
    and measure; any deviation from the prepared bitstring is readout error.
  - GHZ probe: a maximally-entangled state whose ideal measurement histogram is
    a clean 50/50 split between |0...0> and |1...1>; broadening / extra mass on
    other bitstrings flags gate / coherent error on top of readout error.
"""

from __future__ import annotations

from braket.circuits import Circuit


def readout_probe(num_qubits: int) -> Circuit:
    """Identity-prep readout calibration probe (prepare |0...0>, measure)."""
    circ = Circuit()
    for q in range(num_qubits):
        circ.i(q)
    return circ.measure(range(num_qubits))


def ghz_probe(num_qubits: int) -> Circuit:
    """GHZ-state probe: H on q0 then a CNOT ladder, measured.

    Ideal outcome distribution is ~50% |0...0> and ~50% |1...1>.
    """
    if num_qubits < 1:
        raise ValueError("ghz_probe needs at least 1 qubit")
    circ = Circuit().h(0)
    for q in range(num_qubits - 1):
        circ.cnot(q, q + 1)
    return circ.measure(range(num_qubits))


def bell_probe() -> Circuit:
    """Two-qubit Bell-state probe (a GHZ probe specialized to 2 qubits)."""
    return ghz_probe(2)


def probe_circuits(num_qubits: int) -> dict[str, Circuit]:
    """The standard probe set, keyed by name."""
    return {
        "readout": readout_probe(num_qubits),
        "ghz": ghz_probe(num_qubits),
    }
