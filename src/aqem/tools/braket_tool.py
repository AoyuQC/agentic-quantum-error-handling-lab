"""Braket + Mitiq execution tool — a Gateway-shaped seam.

Wraps the vendored ``aqem.braket_mitiq`` primitives behind a small, named API
that the DAG nodes call (``run_probe``, ``calibrate_readout``, ``run_mitigation``).
Keeping these as thin functions makes them straightforward to expose as
AgentCore Gateway MCP tools later (Phase C5) without touching the loop.

The mitigation runner composes any subset of {REM, PT, ZNE}, so the adaptive
loop and the static baseline share identical numerics — only the control policy
differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
from braket.circuits import Circuit
from braket.devices import Device
from mitiq.pt import generate_pauli_twirl_variants
from mitiq.rem import generate_inverse_confusion_matrix
from mitiq.zne import combine_results, construct_circuits

from ..baseline.full_stack import _build_bit_masks, extrapolation_method
from ..braket_mitiq.mitigation_tools import apply_readout_twirl, get_twirled_readout_dist
from ..braket_mitiq.mitiq_braket_tools import braket_rem_twirl_mitigator
from ..braket_mitiq.observable_tools import pauli_grouping
from ..braket_mitiq.program_set_tools import run_with_program_sets
from ..models import Strategy, Technique


def run_probe(circuit: Circuit, device: Device, shots: int) -> dict[str, int]:
    """Run a single probe circuit and return its raw measurement counts."""
    result = device.run(circuit, shots=shots).result()
    return dict(result.measurement_counts)


@dataclass
class ReadoutCalibration:
    """Live (non-serialized) REM calibration carrying the numpy matrix."""

    inverse_confusion_matrix: np.ndarray
    qubit_readout_errors: list[float]
    shots_used: int

    @property
    def quality(self) -> float:
        """A 0..1 quality score: 1 - mean single-qubit readout error."""
        if not self.qubit_readout_errors:
            return 0.0
        return float(max(0.0, 1.0 - np.mean(self.qubit_readout_errors)))


def calibrate_readout(
    num_qubits: int, device: Device, rem_twirls: int, shots: int
) -> ReadoutCalibration:
    """Build the first-order REM inverse confusion matrix via readout twirling."""
    def add_measure(circ: Circuit) -> Circuit:
        return circ.measure(range(num_qubits))

    dist = get_twirled_readout_dist(
        range(num_qubits), rem_twirls, shots=shots, device=device, processor=add_measure
    )
    qubit_errors = [0.0] * num_qubits
    for bitstring, prob in dist.items():
        for n in range(num_qubits):
            if bitstring[n] == "1":
                qubit_errors[n] += prob
    mats = [generate_inverse_confusion_matrix(1, p0=e, p1=e) for e in qubit_errors]
    icm = reduce(np.kron, mats, np.array([[1]]))
    return ReadoutCalibration(icm, [float(e) for e in qubit_errors], shots)


def _program_set_shots(psets) -> int:
    return int(sum(p.total_executables * p.shots_per_executable for p in psets))


@dataclass
class MitigationResult:
    """Output of a mitigation run."""

    value: float
    error_bar: float
    shots_used: int
    zne_data: dict[str, float]
    techniques: list[str]


def run_mitigation(
    circuit: Circuit,
    observable: list[tuple[float, str]],
    device: Device,
    strategy: Strategy,
    calibration: ReadoutCalibration | None = None,
) -> MitigationResult:
    """Run the chosen subset of {REM, PT, ZNE} and return a mitigated estimate.

    Args:
        circuit: the runnable target circuit.
        observable: weighted Pauli terms of the observable to estimate.
        device: Braket device / LocalSimulator.
        strategy: which techniques + parameters to apply.
        calibration: required when the strategy uses REM (the inverse confusion
            matrix); ignored otherwise.

    Returns:
        MitigationResult with the estimate, a jackknife error bar (over twirls),
        the ZNE data points, and the execution shots consumed (REM-calibration
        shots are accounted separately by the caller).
    """
    bases, pauli_terms = pauli_grouping(observable)
    use_rem = strategy.uses(Technique.REM.value)
    use_pt = strategy.uses(Technique.PT.value)
    use_zne = strategy.uses(Technique.ZNE.value)
    K = max(1, strategy.twirl_count)
    scales = strategy.zne_scale_factors if use_zne else [1]

    # 1. Build the (n_scale, K) array of circuit variants.
    base_circuits = construct_circuits(circuit, scale_factors=scales) if use_zne else [circuit]
    arr = np.empty((len(scales), K), dtype=object)
    for si, c in enumerate(base_circuits):
        variants = (
            generate_pauli_twirl_variants(c, num_circuits=K) if use_pt
            else [c.copy() for _ in range(K)]
        )
        for ti in range(K):
            arr[si, ti] = variants[ti]

    # 2. Optional readout twirl + REM correction filter.
    measurement_filter = None
    if use_rem:
        if calibration is None:
            raise ValueError("strategy uses REM but no calibration was provided")
        arr, twirls = apply_readout_twirl(arr)
        bit_masks = _build_bit_masks(twirls, bases)
        measurement_filter = braket_rem_twirl_mitigator(
            calibration.inverse_confusion_matrix, bit_masks=bit_masks
        )

    # 3. Execute via Program Sets.
    shots_per_exec = strategy.shot_per_base * strategy.overhead // K
    result, psets = run_with_program_sets(
        arr, bases, pauli_terms, parameters=[{}], device=device,
        measurement_filter=measurement_filter, shots_per_executable=shots_per_exec,
        return_program_sets=True,
    )

    # 4. Post-process: sum over observables/bases, average over twirls, ZNE.
    twirled = np.sum(result, axis=(2, 3))             # (n_scale, K)
    per_scale = np.sum(twirled, axis=1) / K
    if use_zne and len(scales) >= 2:
        method = extrapolation_method(strategy.zne_factory)
        value = float(combine_results(
            scale_factors=scales, results=per_scale, extrapolation_method=method
        ))
        error_bar = _jackknife_error(twirled, scales, method, K)
    else:
        value = float(per_scale[0])
        error_bar = _sample_error(twirled[0], K)

    techniques = [t for t in (Technique.REM.value, Technique.PT.value, Technique.ZNE.value)
                  if strategy.uses(t)]
    return MitigationResult(
        value=value,
        error_bar=error_bar,
        shots_used=_program_set_shots(psets),
        zne_data={str(sf): float(v) for sf, v in zip(scales, per_scale)},
        techniques=techniques,
    )


def _jackknife_error(twirled: np.ndarray, scales: list[int], method: Any, K: int) -> float:
    """Leave-one-twirl-out jackknife std of the ZNE estimate."""
    if K < 2:
        return 0.0
    by_twirl = twirled.T  # (K, n_scale)
    jack = []
    for i in range(K):
        loo = (np.sum(by_twirl[:i], axis=0) + np.sum(by_twirl[i + 1:], axis=0)) / (K - 1)
        jack.append(combine_results(scale_factors=scales, results=loo, extrapolation_method=method))
    jack = np.array(jack)
    mean = np.average(jack)
    return float(np.sqrt(np.sum(np.square(jack - mean)) * (K - 1) / K**2))


def _sample_error(per_twirl: np.ndarray, K: int) -> float:
    """Standard error of the mean over twirls (no extrapolation case)."""
    if K < 2:
        return 0.0
    return float(np.std(per_twirl, ddof=1) / np.sqrt(K))
