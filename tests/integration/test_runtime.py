"""L/C5 acceptance: the AgentCore Runtime handler runs the loop end-to-end on a
local simulator, persists artifacts locally, and returns the estimate + ledger +
comparison. Runs with the VLM disabled so it is fully offline/deterministic."""

import json

import pytest

from aqem.cloud.runtime import invoke

pytestmark = pytest.mark.integration


def test_runtime_handler_runs_offline(tmp_path):
    res = invoke({
        "qubits": 2,
        "target_accuracy": 0.06,
        "device": "qd_readout_2",
        "use_vlm": False,            # offline: rules-only
        "compare_baseline": True,
        "seed": 7,
        "artifacts": str(tmp_path),
        "run_id": "test",
    })

    assert res["status"] in ("stopped", "exhausted")
    assert res["vlm_used"] is False
    assert res["estimate"] is not None
    assert res["shots_used"] > 0

    # Comparison is present and well-formed (exact shot counts vary because the
    # simulator's sampling is not seedable).
    cmp = res["comparison"]
    assert cmp["adaptive"]["shots"] > 0 and cmp["baseline"]["shots"] > 0
    assert "efficiency_gain_demonstrated" in cmp

    # Artifacts were written locally and are valid JSON.
    audit_path = res["artifacts"]["audit"]
    audit = json.loads(open(audit_path).read())
    assert isinstance(audit, list) and len(audit) > 0
    assert all(r["approved"] for r in audit)  # only approved actions ran


def test_runtime_handler_respects_budget(tmp_path):
    res = invoke({
        "qubits": 2,
        "target_accuracy": 0.0001,   # unmeetable -> exercises budget guard
        "device": "qd_readout_2",
        "use_vlm": False,
        "compare_baseline": False,
        "budget_shots": 80_000,
        "seed": 7,
        "artifacts": str(tmp_path),
        "run_id": "budget",
    })
    assert res["shots_used"] <= 80_000
    assert res["status"] in ("stopped", "exhausted", "failed")
