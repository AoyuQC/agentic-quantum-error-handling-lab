"""Static full-stack Mitiq baseline.

Applies the *complete* mitigation stack — REM (readout error mitigation via
readout twirling + inverse confusion matrix) + Pauli twirling + ZNE — with a
fixed, generous shot budget and no early stopping and no probe. This is the
"run the full stack blindly" reference the adaptive loop is measured against.

The composition mirrors the Braket error-mitigation workflow notebook
(examples/error_mitigation/on_mitiq/4_*.ipynb) and reuses the vendored
primitives in ``aqem.braket_mitiq`` so the only difference vs the adaptive run
is the *control policy*, not the numerics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Any

import numpy as np
from braket.circuits import Circuit
from braket.devices import Device
from mitiq.pt import generate_pauli_twirl_variants
from mitiq.rem import generate_inverse_confusion_matrix
from mitiq.zne import ExpFactory, LinearFactory, PolyFactory, RichardsonFactory, combine_results, construct_circuits

from ..braket_mitiq.mitigation_tools import apply_readout_twirl, get_twirled_readout_dist
from ..braket_mitiq.mitiq_braket_tools import braket_rem_twirl_mitigator
from ..braket_mitiq.observable_tools import pauli_grouping
from ..braket_mitiq.program_set_tools import run_with_program_sets
from ..models import Estimate, Problem

# Extrapolation factories selectable by name (design action: change_zne_factory).
# Each value is a callable taking (scale_factors, exp_values) -> float, as
# expected by ``mitiq.zne.combine_results``.
_FACTORIES = {
    "Linear": LinearFactory.extrapolate,
    "Richardson": RichardsonFactory.extrapolate,
    "Exp": partial(ExpFactory.extrapolate, asymptote=0),
    "Poly": partial(PolyFactory.extrapolate, order=2),
}

# Order in which the adaptive loop escalates factories (design-doc §6.4):
# Linear -> Exp -> Richardson.
FACTORY_ESCALATION = ["Linear", "Exp", "Richardson"]


def extrapolation_method(factory: str, scale_factors: list[int] | None = None):
    """Return a mitiq extrapolation callable for ``combine_results``.

    ``scale_factors`` is accepted for API symmetry but is supplied by
    ``combine_results`` at call time, so it is not needed here.
    """
    if factory not in _FACTORIES:
        raise ValueError(f"Unknown ZNE factory: {factory}. Choose from {list(_FACTORIES)}")
    return _FACTORIES[factory]


@dataclass
class BaselineConfig:
    """Knobs for the static baseline (all fixed up front, no adaptation)."""

    shot_per_base: int = 4000
    overhead: int = 3
    scale_factors: list[int] = field(default_factory=lambda: [1, 3, 7])
    num_twirls: int = 16
    rem_twirls: int = 50
    zne_factory: str = "Exp"


def _build_bit_masks(twirls: np.ndarray, bases: list[str]) -> np.ndarray:
    """Per-twirl, per-basis readout-twirl correction masks (notebook §16)."""
    bit_masks = np.zeros(twirls.shape + (1, len(bases)), dtype=object)
    for n, twirl in np.ndenumerate(twirls):
        for m, b in enumerate(bases):
            new = []
            for tw, base in zip(b, twirl):
                if base == "I":
                    base = "Z"
                new.append("0" if (tw == base or tw == "I" or base == "I") else "1")
            bit_masks[n + (0, m)] = "".join(new)
    return bit_masks


def _readout_inverse_confusion_matrix(
    num_qubits: int, device: Device, rem_twirls: int, shots: int
) -> tuple[np.ndarray, list[float]]:
    """Build the first-order REM inverse confusion matrix from twirled readout."""
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
    return icm, qubit_errors


def _program_set_shots(psets) -> int:
    """Exact shots consumed by a list of ProgramSets."""
    return int(sum(p.total_executables * p.shots_per_executable for p in psets))


def run_full_stack_baseline(
    problem: Problem,
    circuit: Circuit,
    device: Device,
    config: BaselineConfig | None = None,
) -> Estimate:
    """Run the full REM + PT + ZNE stack and return a mitigated estimate.

    Args:
        problem: the estimation task (observable + qubit count).
        circuit: the runnable (bound) target circuit.
        device: a Braket device / LocalSimulator with a noise model.
        config: baseline knobs; defaults to a generous fixed budget.

    Returns:
        Estimate with the mitigated value, a jackknife error bar over twirls,
        the ZNE data points, and the total shots consumed.
    """
    config = config or BaselineConfig()
    nq = problem.num_qubits
    bases, pauli_terms = pauli_grouping(problem.observable)

    # 1. Build ZNE-scaled x Pauli-twirled x readout-twirled circuit variants.
    circuits = np.array(
        [
            generate_pauli_twirl_variants(c, num_circuits=config.num_twirls)
            for c in construct_circuits(circuit, scale_factors=config.scale_factors)
        ],
        dtype=object,
    )
    circuits, twirls = apply_readout_twirl(circuits)
    bit_masks = _build_bit_masks(twirls, bases)

    # 2. REM calibration: inverse confusion matrix via readout twirling.
    rem_shots = config.overhead * config.shot_per_base
    icm, qubit_errors = _readout_inverse_confusion_matrix(nq, device, config.rem_twirls, rem_shots)
    measurement_filter = braket_rem_twirl_mitigator(icm, bit_masks=bit_masks)

    # 3. Execute the variants via Program Sets, REM-correcting each readout.
    shots_per_exec = config.shot_per_base * config.overhead // config.num_twirls
    result, psets = run_with_program_sets(
        circuits,
        bases,
        pauli_terms,
        parameters=[{}],
        device=device,
        measurement_filter=measurement_filter,
        shots_per_executable=shots_per_exec,
        return_program_sets=True,
    )

    # 4. Post-process: sum over observables/bases, average over twirls, ZNE.
    twirled = np.sum(result, axis=(2, 3))            # (n_scale, n_twirls)
    zne_results = np.sum(twirled, axis=1) / config.num_twirls
    method = extrapolation_method(config.zne_factory, config.scale_factors)
    mitigated = float(combine_results(
        scale_factors=config.scale_factors, results=zne_results, extrapolation_method=method
    ))

    # Jackknife error bar over twirls (leave-one-out extrapolations).
    error_bar = _jackknife_error(twirled, config, method)

    exec_shots = _program_set_shots(psets)
    total_shots = exec_shots + rem_shots

    return Estimate(
        value=mitigated,
        error_bar=error_bar,
        shots_used=total_shots,
        techniques=["REM", "PT", "ZNE"],
        zne_data={str(sf): float(v) for sf, v in zip(config.scale_factors, zne_results)},
        metadata={
            "mode": "baseline_full_stack",
            "zne_factory": config.zne_factory,
            "num_twirls": config.num_twirls,
            "rem_twirls": config.rem_twirls,
            "qubit_readout_errors": [float(e) for e in qubit_errors],
            "execution_shots": exec_shots,
            "rem_shots": rem_shots,
            "noisy_sf1": float(zne_results[0]),
        },
    )


def _jackknife_error(twirled: np.ndarray, config: BaselineConfig, method: Any) -> float:
    """Leave-one-twirl-out jackknife standard error of the ZNE estimate."""
    n = config.num_twirls
    if n < 2:
        return 0.0
    by_scale = twirled.T  # (n_twirls, n_scale)
    jackknife = []
    for i in range(n):
        loo = (np.sum(by_scale[:i, :], axis=0) + np.sum(by_scale[i + 1:, :], axis=0)) / (n - 1)
        jackknife.append(
            combine_results(
                scale_factors=config.scale_factors, results=loo, extrapolation_method=method
            )
        )
    jackknife = np.array(jackknife)
    mean = np.average(jackknife)
    return float(np.sqrt(np.sum(np.square(jackknife - mean)) * (n - 1) / n**2))
