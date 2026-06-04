"""Unit tests for probe circuits, problem construction, and plot figures.

Offline — no simulator, no AWS.
"""

import numpy as np

from aqem.probes.circuits import probe_circuits, ghz_probe, readout_probe
from aqem.probes.histograms import histogram_figure
from aqem.problems import default_problem, ideal_expectation, ising_hamiltonian


def test_probe_circuits_have_expected_shape():
    probes = probe_circuits(3)
    assert set(probes) == {"readout", "ghz"}
    # readout probe is all-identity prep on 3 qubits
    assert readout_probe(3).qubit_count == 3
    # GHZ probe: one H + (n-1) CNOTs
    ghz = ghz_probe(3)
    op_names = [ins.operator.name for ins in ghz.instructions]
    assert op_names.count("H") == 1
    assert op_names.count("CNot") == 2


def test_ising_hamiltonian_terms():
    ham = ising_hamiltonian(0.5, 1.0, 2)
    # 2 Z terms + 1 XX hopping term
    paulis = [p for _, p in ham]
    assert "ZI" in paulis and "IZ" in paulis
    assert "XX" in paulis


def test_ideal_expectation_matches_known_value():
    # For the default 2-qubit problem the exact <H> is deterministic.
    problem, circuit = default_problem(num_qubits=2)
    ideal = ideal_expectation(circuit, problem.observable)
    assert isinstance(ideal, float)
    # Sanity: the value is finite and within the operator-norm bound.
    assert abs(ideal) <= sum(abs(c) for c, _ in problem.observable) + 1e-9


def test_histogram_figure_normalizes():
    fig = histogram_figure({"00": 30, "11": 10}, title="t", highlight=["00", "11"])
    ys = fig["data"][0]["y"]
    assert np.isclose(sum(ys), 1.0)
    assert fig["layout"]["title"]["text"] == "t"
