"""Unit tests for the deterministic Policy gates (no simulator)."""

from aqem.models import Budget
from aqem.policy import Action, ActionRequest, AuditLog, Policy


def _policy(shots_total=100_000, max_retries=3):
    return Policy(Budget(shots_total=shots_total), AuditLog(), max_retries_per_node=max_retries)


def test_action_set_gate_rejects_unknown_action():
    policy = _policy()
    req = ActionRequest(action="hack_the_qpu", node_id="execute", predicted_shots=10)
    decision = policy.check(req)
    assert not decision.approved
    assert "controlled action set" in decision.reason


def test_budget_hard_gate_rejects_overrun():
    policy = _policy(shots_total=1000)
    req = ActionRequest(action=Action.RUN_ZNE_SWEEP, node_id="execute", predicted_shots=5000)
    decision = policy.check(req)
    assert not decision.approved
    assert "budget exceeded" in decision.reason


def test_budget_gate_allows_within_budget_then_charges():
    policy = _policy(shots_total=1000)
    req = ActionRequest(action=Action.RUN_READOUT_MITIGATION, node_id="execute", predicted_shots=600)
    assert policy.check(req).approved
    policy.charge(shots=600)
    # A second 600-shot action would now overrun.
    assert not policy.check(req).approved


def test_no_recalibration_guard():
    policy = _policy()
    req = ActionRequest(
        action=Action.RUN_READOUT_CONFUSION_MATRIX,
        node_id="x",
        params={"mode": "recalibrate_device"},
        predicted_shots=10,
    )
    decision = policy.check(req)
    assert not decision.approved
    assert "recalibration" in decision.reason or "forbidden" in decision.reason


def test_retry_cap_gate():
    policy = _policy(max_retries=2)
    for _ in range(3):
        policy.record_retry("execute")  # 3 retries > cap of 2
    req = ActionRequest(action=Action.INCREASE_SHOTS, node_id="execute", predicted_shots=10)
    decision = policy.check(req)
    assert not decision.approved
    assert "retry cap" in decision.reason


def test_every_check_is_audited():
    policy = _policy()
    policy.check(ActionRequest(action=Action.STOP_AND_REPORT, node_id="report"))
    policy.check(ActionRequest(action="bogus", node_id="x"))
    assert len(policy.audit) == 2
    assert len(policy.audit.approved()) == 1
    assert len(policy.audit.rejected()) == 1
