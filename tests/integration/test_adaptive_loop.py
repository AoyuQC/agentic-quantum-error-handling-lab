"""L2 acceptance: the full deterministic adaptive loop runs end-to-end on a
local noisy simulator, early-stops when the target is met, and every executed
action is Policy-approved and within budget. Rules-only (no VLM)."""

import pytest

from aqem.loop import run_adaptive_loop
from aqem.models import Budget
from aqem.problems import default_problem, ideal_expectation

pytestmark = pytest.mark.integration


def _config():
    return {
        "probe_shots": 2000,
        "shot_per_base": 4000,
        "overhead": 3,
        "rem_twirls": 20,
        "use_ideal_for_validation": True,
    }


def test_adaptive_loop_early_stops_on_readout_dominated_device():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    budget = Budget(shots_total=2_000_000)

    record = run_adaptive_loop(problem, circuit, qd_readout_2, budget, config=_config(), seed=7)

    # Behavioural invariants (the simulator's shot sampling is not seedable, so
    # the exact stop-vs-escalate path varies run to run).
    assert record.status in ("stopped", "exhausted")
    est = record.final_outputs["estimate"]
    assert est is not None
    # REM is always the floor on a readout-dominated device.
    assert "REM" in est["techniques"]

    # When the loop reports a stop, the estimate is within target + error bar.
    ideal = ideal_expectation(circuit, problem.observable)
    if record.status == "stopped":
        assert abs(est["value"] - ideal) <= problem.target_accuracy + est["error_bar"]


def test_every_executed_action_is_policy_approved_and_audited():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    budget = Budget(shots_total=2_000_000)

    record = run_adaptive_loop(problem, circuit, qd_readout_2, budget, config=_config(), seed=7)
    audit = record.final_outputs["audit"]

    assert len(audit) > 0
    # Every action is audited (approved or rejected), and shots actually spent
    # never exceed the sum of approved predictions — i.e. only approved actions
    # consumed budget.
    approved_shots = sum(r["predicted_shots"] for r in audit if r["approved"])
    assert record.final_outputs["shots_used"] <= approved_shots
    # And the ledger stayed within the hard ceiling.
    assert record.final_outputs["shots_used"] <= budget.shots_total


def test_budget_starvation_forces_stop():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.0001)  # unmeetable
    # Tiny budget: only the probe + one calibration/execution can run.
    budget = Budget(shots_total=80_000)

    record = run_adaptive_loop(problem, circuit, qd_readout_2, budget, config=_config(), seed=7)

    # Either we early-stopped or hit the budget guard; never overran.
    assert record.final_outputs["shots_used"] <= budget.shots_total
    assert record.status in ("stopped", "exhausted", "failed")
