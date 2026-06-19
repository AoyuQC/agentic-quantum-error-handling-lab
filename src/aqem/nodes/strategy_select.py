"""strategy_select node — choose the minimal mitigation recipe.

Rules-first: maps the probe classification to a minimal Strategy. On a
RETRY_STRATEGY re-run, escalates the previous strategy (add ZNE -> add PT ->
advance ZNE factory) instead of restarting from minimal.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..decision.rules import escalate_strategy, select_strategy
from ..models import NodeResult, Strategy


class StrategySelectNode(Node):
    node_id = "strategy_select"
    dependencies = ("empirical_probe",)
    # Re-running strategy invalidates the whole downstream chain.
    invalidates = ("readout_calibrate", "circuit_generate", "execute", "post_process")

    def run(self, ctx: RunContext) -> NodeResult:
        classification = ctx.get("empirical_probe").get("classification", {})
        target = ctx.problem.target_accuracy

        # If the orchestration agent proposed the next parameter set, apply it
        # verbatim (the agent is the decider). Consume it so a later rules-driven
        # retry doesn't reuse a stale proposal.
        agent_strategy = ctx.config.pop("_agent_strategy", None)
        prior = ctx.config.get("_last_strategy")
        if agent_strategy is not None:
            strategy = Strategy.from_dict(agent_strategy)
            source = "agent"
        # If a prior strategy exists (this is an escalation retry), strengthen it.
        elif prior is not None:
            strategy = escalate_strategy(Strategy.from_dict(prior))
            source = "escalation"
        else:
            strategy = select_strategy(
                dominant_error=classification.get("dominant_error", "readout"),
                target_accuracy=target,
                suggested_focus=classification.get("suggested_focus"),
            )
            source = "rules"

        # Carry shot knobs from config — except when the agent set them: the
        # agent owns the full parameter set, so its shot_per_base/overhead win.
        if source != "agent":
            strategy.shot_per_base = int(ctx.config.get("shot_per_base", strategy.shot_per_base))
            strategy.overhead = int(ctx.config.get("overhead", strategy.overhead))
            strategy.rem_twirls = int(ctx.config.get("rem_twirls", strategy.rem_twirls))

        ctx.config["_last_strategy"] = strategy.to_dict()
        outputs = {"strategy": strategy.to_dict(), "source": source}
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs)
