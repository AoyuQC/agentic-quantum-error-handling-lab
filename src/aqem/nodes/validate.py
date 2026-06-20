"""validate node — decide STOP vs a RETRY_* mode (rules-first).

Compares the estimate's error to the target and produces a :class:`Decision`
the engine acts on. In Phase L2 this is purely the deterministic ``decide``
rules; Phase L3 passes a VLM verdict on the extrapolation plot as an additional,
confidence-gated input.

When benchmarking locally we know the exact reference, so the *true* error is
used as the stopping metric; otherwise the error bar is the proxy.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..decision.rules import decide
from ..models import Decision, DecisionAction, Estimate, NodeResult, Strategy  # noqa: F401
from ..problems import ideal_expectation
from ._stream import token_emitter


class ValidateNode(Node):
    node_id = "validate"
    dependencies = ("post_process",)
    invalidates = ()  # the engine decides what to invalidate from the Decision

    def run(self, ctx: RunContext) -> NodeResult:
        estimate = Estimate.from_dict(ctx.get("post_process")["estimate"])
        strategy = Strategy.from_dict(ctx.get("strategy_select")["strategy"])
        target = ctx.problem.target_accuracy

        # True error when a cheap exact reference is available (local benchmarking).
        error_estimate = None
        if ctx.config.get("use_ideal_for_validation", True):
            ideal = ideal_expectation(ctx.circuit, ctx.problem.observable)
            error_estimate = abs(estimate.value - ideal)

        confidence_threshold = float(ctx.config.get("vlm_confidence_threshold", 0.5))

        # The ZNE extrapolation plot the validate step inspects (only meaningful
        # when ZNE ran, i.e. there are >=2 scale points). Build it once and both
        # show it in the UI and hand it to the VLM.
        plots: list = []
        zne_fig = None
        if len(estimate.zne_data) >= 2:
            from ..reporting.plots import zne_extrapolation_figure

            zne_fig = zne_extrapolation_figure(estimate.zne_data, estimate.value)
            plots.append({"name": "zne_extrapolation", "format": "plotly", "data": zne_fig})

        # VLM judgment on the extrapolation plot (analysis tool; never decides).
        vlm_verdict = None      # the confidence-gated verdict the decision uses
        vlm_trace = None        # the full verdict (incl. image/prompt/raw) for the UI
        tools = ctx.tool_client()
        if tools.vlm_enabled and zne_fig is not None:
            verdict = tools.validate(
                [{"name": "zne_extrapolation", "format": "plotly", "data": zne_fig}],
                confidence_threshold,
                on_token=token_emitter(ctx, self.node_id, "vlm"),
            )
            vlm_trace = verdict
            if not verdict.get("degraded"):
                vlm_verdict = verdict

        # The orchestration agent is the decider; the deterministic rules are
        # the fallback. Record this attempt into the running history first so
        # the agent can reason over the trajectory (and so we can detect a
        # plateau). The VLM is demoted to an analysis tool: its verdict is an
        # *input* to the agent and may steer a retry, but never ends the loop.
        metric = error_estimate if error_estimate is not None else estimate.error_bar
        attempts: list = list(ctx.config.get("_attempts", []))
        attempts.append({
            "iteration": len(attempts) + 1,
            "techniques": list(strategy.techniques),
            "zne_factory": strategy.zne_factory,
            "zne_scale_factors": list(strategy.zne_scale_factors),
            "error": error_estimate,
            "error_bar": estimate.error_bar,
            "shots_used": ctx.policy.budget.shots_used,
        })
        ctx.config["_attempts"] = attempts

        decision: Decision = None  # type: ignore[assignment]
        agent_strategy = None
        # Target met is a deterministic STOP that needs no agent call.
        if metric <= target:
            decision = Decision(
                action=DecisionAction.STOP.value,
                reason=f"target met: {metric:.4f} <= {target:.4f}",
                source="rules",
            )
        elif tools.vlm_enabled:
            from ..decision.agent import propose_decision

            proposed = propose_decision(
                ctx.vlm,
                target=target,
                current_error=error_estimate,
                current_error_bar=estimate.error_bar,
                current_strategy=strategy,
                attempts=attempts,
                vlm_analysis=vlm_verdict,
                remaining_shots=ctx.policy.budget.remaining_shots(),
                iteration=len(attempts),
                max_iterations=int(ctx.config.get("max_iterations", 8)),
                confidence_threshold=confidence_threshold,
                on_token=token_emitter(ctx, self.node_id, "agent"),
            )
            if proposed is not None:
                decision, agent_strategy = proposed
                if agent_strategy is not None:
                    # The strategy_select node applies this on the retry re-run.
                    ctx.config["_agent_strategy"] = agent_strategy.to_dict()

        # Fallback: deterministic rules (no agent, agent abstained/low-conf, or
        # VLM disabled). The VLM verdict may still steer *which* retry to run.
        if decision is None:
            decision = decide(
                error_bar=estimate.error_bar,
                error_estimate=error_estimate,
                target_accuracy=target,
                strategy=strategy,
                vlm_verdict=vlm_verdict,
                confidence_threshold=confidence_threshold,
            )

        # Guard: if we're out of budget for any further runs, stop and report.
        if decision.action != DecisionAction.STOP.value:
            if ctx.policy.budget.remaining_shots() is not None and \
                    ctx.policy.budget.remaining_shots() <= 0:
                decision = Decision(
                    action=DecisionAction.STOP.value,
                    reason="budget exhausted; stopping with best estimate",
                    source="rules",
                )

        outputs = {
            "decision": decision.to_dict(),
            "error_estimate": error_estimate,
            "metric_value": error_estimate if error_estimate is not None else estimate.error_bar,
            # Full VLM verdict (rationale, confidence, the ZNE image it saw, the
            # prompt, the raw answer) so the UI can show the agent's reasoning.
            "vlm_verdict": vlm_trace,
        }
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs, plots=plots)
