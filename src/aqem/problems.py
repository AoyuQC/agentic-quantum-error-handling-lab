"""Problem construction and the exact (noiseless) reference.

Provides a small transverse-field-Ising-style estimation task and the helpers
to build its circuit, its observable matrix, and the exact expectation value
(via a noiseless statevector) used as ground truth for the efficiency
comparison. Mirrors the construction in the Braket error-mitigation workflow
notebook (examples/error_mitigation/on_mitiq/4_*.ipynb).
"""

from __future__ import annotations

from functools import reduce

import numpy as np
from braket.circuits import Circuit
from braket.quantum_information import PauliString

from .models import Problem


def ising_hamiltonian(
    hopping: float, self_interaction: float, num_qubits: int
) -> list[tuple[float, str]]:
    """Weighted Pauli terms for a 1D Ising Hamiltonian.

    Z self-interaction on each qubit plus XX hopping between neighbors —
    the same form used in the reference workflow notebook.
    """
    n = num_qubits
    hamiltonian: list[tuple[float, str]] = []
    for i in range(num_qubits):
        hamiltonian.append((self_interaction, i * "I" + "Z" + (n - i - 1) * "I"))
        if i > 0:
            hamiltonian.append((hopping, (i - 1) * "I" + "XX" + (n - i - 1) * "I"))
    return hamiltonian


def ansatz_circuit(num_qubits: int, theta1: float = 0.3, theta2: float = 0.2) -> Circuit:
    """A shallow layered ansatz: Rx — CZ (even pairs) — Rx — CZ (odd pairs).

    Parameters are bound to concrete angles so the circuit is directly
    runnable (no free parameters).
    """
    circ = Circuit()
    for i in range(num_qubits):
        circ.rx(i, theta1)
    for i in range(0, num_qubits - 1, 2):
        circ.cz(i, i + 1)
    for i in range(num_qubits):
        circ.rx(i, theta2)
    for i in range(1, num_qubits - 1, 2):
        circ.cz(i, i + 1)
    return circ


def observable_matrix(observable: list[tuple[float, str]]) -> np.ndarray:
    """Dense matrix of a weighted Pauli-term observable."""
    return reduce(
        np.add,
        [
            c * PauliString(p).to_unsigned_observable(include_trivial=True).to_matrix()
            for c, p in observable
        ],
    )


def ideal_expectation(circuit: Circuit, observable: list[tuple[float, str]]) -> float:
    """Exact <psi|O|psi> from the noiseless statevector of ``circuit``.

    This is the ground-truth reference for accuracy comparisons; it does not
    consume any shot budget (it is a deterministic linear-algebra evaluation,
    not a sampled run).
    """
    dim = 2 ** _circuit_qubit_count(circuit, observable)
    statevector = circuit.to_unitary()[:, 0].reshape((dim, 1))
    matrix = observable_matrix(observable)
    value = (np.conj(statevector).T @ matrix @ statevector)[0, 0]
    return float(value.real)


def _circuit_qubit_count(circuit: Circuit, observable: list[tuple[float, str]]) -> int:
    # Prefer the observable's Pauli-string length (covers idle qubits the
    # circuit's unitary might not expand); fall back to the circuit.
    if observable:
        return len(observable[0][1])
    return circuit.qubit_count


def default_problem(num_qubits: int = 2, target_accuracy: float = 0.05) -> tuple[Problem, Circuit]:
    """Build the default Ising estimation task and its bound circuit.

    Returns:
        (problem, circuit): the :class:`Problem` spec and the runnable circuit.
    """
    observable = ising_hamiltonian(0.5, 1.0, num_qubits)
    circuit = ansatz_circuit(num_qubits)
    problem = Problem(
        num_qubits=num_qubits,
        observable=observable,
        target_accuracy=target_accuracy,
        description=f"{num_qubits}-qubit transverse-field Ising <H> estimate",
    )
    return problem, circuit
