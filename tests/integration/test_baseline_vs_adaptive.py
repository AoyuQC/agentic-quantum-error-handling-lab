"""L4 acceptance: on a readout-dominated noise model, the adaptive loop reaches
the target accuracy with materially fewer shots than the blind full-stack
baseline — the headline efficiency claim — and the CLI commands run."""

import json

import pytest

from aqem.baseline.full_stack import BaselineConfig, run_full_stack_baseline
from aqem.loop import run_adaptive_loop
from aqem.models import Budget, Estimate
from aqem.problems import default_problem, ideal_expectation
from aqem.reporting.efficiency import compare

pytestmark = pytest.mark.integration


def test_adaptive_beats_baseline_on_readout_dominated_device():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    ideal = ideal_expectation(circuit, problem.observable)

    # Blind full stack: REM + PT + ZNE with a generous fixed budget, no early stop.
    baseline_est = run_full_stack_baseline(
        problem, circuit, qd_readout_2,
        BaselineConfig(shot_per_base=4000, overhead=3, scale_factors=[1, 3, 7],
                       num_twirls=8, rem_twirls=20, zne_factory="Exp"),
    )

    # Adaptive loop: probe -> minimal strategy -> early stop.
    record = run_adaptive_loop(
        problem, circuit, qd_readout_2, Budget(shots_total=2_000_000),
        config={"probe_shots": 2000, "shot_per_base": 4000, "overhead": 3,
                "rem_twirls": 20, "use_ideal_for_validation": True},
        seed=7,
    )
    adaptive_est = Estimate.from_dict(record.final_outputs["estimate"])

    cmp = compare(adaptive_est, baseline_est, ideal, problem.target_accuracy)

    # The headline claim: adaptive hits the target with strictly fewer shots,
    # at accuracy no worse than the baseline.
    assert cmp.adaptive_meets_target
    assert cmp.adaptive.shots < cmp.baseline.shots
    assert cmp.efficiency_gain_demonstrated


def test_cli_report_runs_and_emits_json(capsys):
    from aqem.cli import main

    rc = main([
        "report", "--device", "qd_readout_2", "--target", "0.06",
        "--seed", "7", "--json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Efficiency comparison" in out
    # The JSON blob after the marker parses and carries the verdict.
    from aqem.cli import _JSON_MARKER

    blob = out.split(_JSON_MARKER, 1)[1]
    data = json.loads(blob)
    assert "efficiency_gain_demonstrated" in data
    assert data["adaptive"]["shots"] < data["baseline"]["shots"]


def test_cli_baseline_and_run_commands(capsys):
    from aqem.cli import main

    assert main(["baseline", "--device", "qd_readout_2", "--target", "0.06"]) == 0
    assert "baseline" in capsys.readouterr().out

    assert main(["run", "--device", "qd_readout_2", "--target", "0.06", "--seed", "7"]) == 0
    assert "adaptive" in capsys.readouterr().out
