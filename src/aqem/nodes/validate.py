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
from ..models import Decision, DecisionAction, Estimate, NodeResult, Strategy
from ..problems import ideal_expectation
from ..tools.vlm_tool import validate_with_vlm


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

        # VLM judgment on the ZNE extrapolation plot (only meaningful when ZNE
        # ran, i.e. there are >=2 scale points to inspect).
        vlm_verdict = None      # the confidence-gated verdict the decision uses
        vlm_trace = None        # the full verdict (incl. image/prompt/raw) for the UI
        if ctx.vlm is not None and len(estimate.zne_data) >= 2:
            from ..reporting.plots import zne_extrapolation_figure

            fig = zne_extrapolation_figure(estimate.zne_data, estimate.value)
            verdict = validate_with_vlm(
                ctx.vlm,
                [{"name": "zne_extrapolation", "format": "plotly", "data": fig}],
                confidence_threshold,
            )
            vlm_trace = verdict
            if not verdict.get("degraded"):
                vlm_verdict = verdict

        decision: Decision = decide(
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
        return NodeResult(node_id=self.node_id, outputs=outputs)
