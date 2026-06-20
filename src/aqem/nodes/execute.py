"""execute node — run the mitigation under the Policy cost gate.

Predicts the execution shots from the circuit plan + strategy, asks Policy for
approval (the budget hard gate), and only then spends them via the shared
``run_mitigation`` runner. On RETRY_SHOTS the per-basis shots are increased.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..models import NodeResult, Strategy, Technique
from ..policy import Action, ActionRequest
from ..tools.braket_tool import MitigationResult


def _action_for(strategy: Strategy) -> Action:
    """Pick the controlled action that best labels this execution."""
    if strategy.uses(Technique.ZNE.value):
        return Action.RUN_ZNE_SWEEP
    if strategy.uses(Technique.PT.value):
        return Action.RUN_PAULI_TWIRLING
    return Action.RUN_READOUT_MITIGATION


class ExecuteNode(Node):
    node_id = "execute"
    dependencies = ("circuit_generate",)
    invalidates = ("post_process",)

    def run(self, ctx: RunContext) -> NodeResult:
        strategy = Strategy.from_dict(ctx.get("strategy_select")["strategy"])

        # More shots-per-base on a shots retry (adaptive escalation).
        retries = ctx.policy.retry_count(self.node_id)
        if retries:
            strategy.shot_per_base = strategy.shot_per_base * (retries + 1)

        plan = ctx.get("circuit_generate")["plan"]
        # Program-set executables each get shot_per_base*overhead//twirls shots.
        per_exec = strategy.shot_per_base * strategy.overhead // max(1, strategy.twirl_count)
        predicted = per_exec * plan["n_executables"]

        req = ActionRequest(
            action=_action_for(strategy),
            node_id=self.node_id,
            params={"shot_per_base": strategy.shot_per_base, "techniques": strategy.techniques},
            predicted_shots=predicted,
        )
        decision = ctx.policy.check(req)
        if not decision.approved:
            return NodeResult(node_id=self.node_id, status="failed", error=decision.reason)

        calibration = None
        if strategy.uses(Technique.REM.value):
            calibration = ctx.get("readout_calibrate").get("_live")
            if calibration is None:
                return NodeResult(
                    node_id=self.node_id, status="failed",
                    error="REM strategy but readout_calibrate produced no calibration",
                )

        result: MitigationResult = ctx.tool_client().run_mitigation(
            ctx.circuit, ctx.problem.observable, strategy, calibration
        )
        ctx.policy.charge(shots=result.shots_used)

        outputs = {
            "value": result.value,
            "error_bar": result.error_bar,
            "shots_used": result.shots_used,
            "zne_data": result.zne_data,
            "techniques": result.techniques,
            # How many circuits actually ran this pass (scales x twirls x bases),
            # so the execute card can label the folded set shown from
            # circuit_generate (the per-iteration difference).
            "n_executables": ctx.get("circuit_generate").get("plan", {}).get("n_executables"),
        }
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs, shots_used=result.shots_used)
