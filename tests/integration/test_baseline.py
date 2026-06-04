"""L1 acceptance: the static full-stack baseline runs on a local noisy
simulator, produces an estimate + shot count, and the efficiency harness scores
it against the exact noiseless reference.
"""

import pytest

from aqem.baseline.full_stack import BaselineConfig, run_full_stack_baseline
from aqem.problems import default_problem, ideal_expectation
from aqem.reporting.efficiency import accuracy_point

pytestmark = pytest.mark.integration


def test_full_stack_baseline_produces_estimate_and_shotcount():
    from aqem.braket_mitiq.noise_models import qd_total

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.1)
    # Small but real run to keep CI fast.
    config = BaselineConfig(
        shot_per_base=2000, overhead=3, scale_factors=[1, 3],
        num_twirls=2, rem_twirls=2, zne_factory="Exp",
    )

    estimate = run_full_stack_baseline(problem, circuit, qd_total, config)

    assert estimate.shots_used > 0
    assert estimate.techniques == ["REM", "PT", "ZNE"]
    assert set(estimate.zne_data) == {"1", "3"}
    # execution + REM shots accounted separately and sum to the total
    md = estimate.metadata
    assert md["execution_shots"] + md["rem_shots"] == estimate.shots_used


def test_baseline_scored_against_noiseless_reference():
    from aqem.braket_mitiq.noise_models import qd_total

    problem, circuit = default_problem(num_qubits=2)
    ideal = ideal_expectation(circuit, problem.observable)

    config = BaselineConfig(
        shot_per_base=4000, overhead=3, scale_factors=[1, 3],
        num_twirls=4, rem_twirls=4, zne_factory="Exp",
    )
    estimate = run_full_stack_baseline(problem, circuit, qd_total, config)
    point = accuracy_point("baseline", estimate, ideal)

    assert point.shots == estimate.shots_used
    # Full-stack mitigation should land within a loose tolerance of the truth.
    assert point.error < 0.5
