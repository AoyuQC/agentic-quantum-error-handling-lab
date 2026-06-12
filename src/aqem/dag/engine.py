"""Deterministic DAG state machine with cascading invalidation.

The engine executes nodes in topological order. The ``validate`` node may emit a
retry :class:`~aqem.models.Decision`; the engine then invalidates the affected
sub-DAG (transitively, following each node's ``invalidates`` edges), re-runs the
dirty nodes in order, and re-validates — until the decision is STOP or a retry
budget is exhausted.

DAG validation (acyclicity + known dependencies) adapts the cycle-detection
approach from the NVIDIA blueprint's ``tools/workflow_tool.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import Decision, DecisionAction, NodeResult
from .context import RunContext
from .node import Node


def _validate_dag(nodes: dict[str, Node]) -> None:
    """Raise ValueError if dependencies are unknown or the graph has a cycle."""
    # All declared dependencies must exist.
    for nid, node in nodes.items():
        for dep in node.dependencies:
            if dep not in nodes:
                raise ValueError(f"node '{nid}' depends on unknown node '{dep}'")

    # Cycle detection via DFS coloring (white/grey/black).
    WHITE, GREY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in nodes}

    def visit(nid: str) -> None:
        color[nid] = GREY
        for dep in nodes[nid].dependencies:
            if color[dep] == GREY:
                raise ValueError(f"dependency cycle detected at '{nid}' -> '{dep}'")
            if color[dep] == WHITE:
                visit(dep)
        color[nid] = BLACK

    for nid in nodes:
        if color[nid] == WHITE:
            visit(nid)


def _topological_order(nodes: dict[str, Node]) -> list[str]:
    """Return node ids in dependency order (deps before dependents)."""
    order: list[str] = []
    visited: set[str] = set()

    def visit(nid: str) -> None:
        if nid in visited:
            return
        for dep in nodes[nid].dependencies:
            visit(dep)
        visited.add(nid)
        order.append(nid)

    for nid in nodes:
        visit(nid)
    return order


def _transitive_invalidation(nodes: dict[str, Node], seeds: list[str]) -> set[str]:
    """All nodes made stale by re-running ``seeds`` (follow ``invalidates`` edges)."""
    dirty: set[str] = set()
    frontier = list(seeds)
    while frontier:
        nid = frontier.pop()
        if nid in dirty or nid not in nodes:
            continue
        dirty.add(nid)
        frontier.extend(nodes[nid].invalidates)
    return dirty


@dataclass
class RunRecord:
    """Outcome of a full engine run."""

    status: str                                  # "stopped" | "exhausted" | "failed"
    decision: Optional[Decision] = None
    iterations: int = 0
    node_results: list[NodeResult] = field(default_factory=list)
    final_outputs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": self.decision.to_dict() if self.decision else None,
            "iterations": self.iterations,
            "node_results": [r.to_dict() for r in self.node_results],
            "final_outputs": self.final_outputs,
            "reason": self.reason,
        }


# Which nodes a retry decision invalidates. The engine seeds the cascade from
# these and follows each node's ``invalidates`` edges transitively.
_RETRY_SEEDS = {
    DecisionAction.RETRY_SHOTS.value: ["execute"],
    DecisionAction.RETRY_CALIBRATION.value: ["readout_calibrate"],
    DecisionAction.RETRY_STRATEGY.value: ["strategy_select"],
}


class DAGEngine:
    """Executes a set of nodes with retry-driven cascading invalidation."""

    def __init__(
        self,
        nodes: list[Node],
        validate_node_id: str = "validate",
        terminal_node_id: str | None = "report",
        max_iterations: int = 8,
        observer: "Callable[[dict], None] | None" = None,
    ):
        self.nodes: dict[str, Node] = {n.node_id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node_id among nodes")
        _validate_dag(self.nodes)
        self.order = _topological_order(self.nodes)
        self.validate_node_id = validate_node_id
        # The terminal node runs exactly once, after the loop concludes (it is
        # excluded from the iterative dirty-node loop).
        self.terminal_node_id = terminal_node_id if terminal_node_id in self.nodes else None
        self.max_iterations = max_iterations
        # Optional progress observer: called with small JSON-able event dicts
        # ({"event": ..., ...}) as the run proceeds. Used by the web UI to
        # stream live node/decision progress; a no-op by default.
        self.observer = observer

    def _emit(self, event: dict[str, Any]) -> None:
        if self.observer is not None:
            try:
                self.observer(event)
            except Exception:  # never let UI plumbing break a run
                pass

    def invalidation_set(self, action: str) -> set[str]:
        """Public helper: nodes invalidated by a given retry action (for tests)."""
        seeds = _RETRY_SEEDS.get(action, [])
        return _transitive_invalidation(self.nodes, seeds)

    def run(self, ctx: RunContext) -> RunRecord:
        """Run the pipeline, applying retries until STOP or budget exhaustion."""
        record = RunRecord(status="failed")
        # First pass: everything except the terminal node is dirty.
        loop_nodes = {nid for nid in self.nodes if nid != self.terminal_node_id}
        dirty: set[str] = set(loop_nodes)
        # Snapshot of the last successful post_process, so a failed retry can
        # still report the best estimate obtained so far.
        last_post_process: dict[str, Any] = {}

        self._emit({"event": "run_start", "nodes": list(self.order)})

        for iteration in range(1, self.max_iterations + 1):
            record.iterations = iteration
            self._emit({"event": "iteration", "iteration": iteration})

            # Execute dirty (non-terminal) nodes in topological order.
            for nid in self.order:
                if nid not in dirty or nid == self.terminal_node_id:
                    continue
                self._emit({"event": "node_start", "node": nid, "iteration": iteration})
                result = self.nodes[nid].run(ctx)
                record.node_results.append(result)
                self._emit({
                    "event": "node_done", "node": nid, "iteration": iteration,
                    "status": result.status, "shots_used": result.shots_used,
                    # Surface what the node concluded so the UI can show the
                    # agent's reasoning. ``counts`` holds large histogram arrays
                    # (rendered separately as figures) and ``_``-prefixed keys
                    # are internal (e.g. live calibration objects), so drop both.
                    "detail": {
                        k: v for k, v in result.outputs.items()
                        if k != "counts" and not k.startswith("_")
                    },
                    # The figures this node produced — what the agent "sees".
                    "plots": list(result.plots),
                })
                if result.status == "failed":
                    record.status = "failed"
                    record.reason = f"node '{nid}' failed: {result.error}"
                    # Restore the last good estimate so report can still run.
                    if not ctx.has("post_process") and last_post_process:
                        ctx.put("post_process", last_post_process)
                    self._emit({"event": "node_failed", "node": nid, "reason": result.error})
                    return self._finalize(ctx, record)
                if nid == "post_process":
                    last_post_process = ctx.get("post_process")

            # Read the validate node's decision.
            decision = self._decision_from(ctx)
            record.decision = decision
            # The validate node stores the metric it compared to target; surface
            # it (with the target) so the UI can show "err 0.0013 <= target 0.06".
            validate_out = ctx.get(self.validate_node_id) if ctx.has(self.validate_node_id) else {}
            self._emit({
                "event": "decision", "iteration": iteration,
                "action": decision.action, "reason": decision.reason,
                "source": decision.source,
                "metric_value": validate_out.get("metric_value"),
                "error_estimate": validate_out.get("error_estimate"),
                "target": ctx.problem.target_accuracy,
            })

            if decision.action == DecisionAction.STOP.value:
                record.status = "stopped"
                record.reason = decision.reason
                self._emit({"event": "run_end", "status": "stopped", "reason": decision.reason})
                return self._finalize(ctx, record)

            # A retry: count it against the seed node's retry budget.
            seeds = _RETRY_SEEDS.get(decision.action, [])
            for seed in seeds:
                ctx.policy.record_retry(seed)

            # Compute the dirty set and clear those artifacts so they recompute.
            dirty = self.invalidation_set(decision.action)
            # validate must re-run too so the loop re-evaluates after recompute.
            dirty.add(self.validate_node_id)
            ctx.clear(list(dirty))

        record.status = "exhausted"
        record.reason = f"max_iterations ({self.max_iterations}) reached without STOP"
        return self._finalize(ctx, record)

    def _finalize(self, ctx: RunContext, record: RunRecord) -> RunRecord:
        """Run the terminal (report) node once and capture final outputs.

        If no estimate was ever produced (e.g. the very first execute was
        budget-rejected), emit a minimal report so callers always get a
        consistent ``shots_used`` / ``audit`` payload.
        """
        if self.terminal_node_id and ctx.has("post_process"):
            result = self.nodes[self.terminal_node_id].run(ctx)
            record.node_results.append(result)
            record.final_outputs = ctx.get(self.terminal_node_id)
        else:
            record.final_outputs = {
                "estimate": None,
                "shots_used": ctx.policy.budget.shots_used,
                "audit": ctx.policy.audit.records,
                "decision": record.decision.to_dict() if record.decision else None,
            }
        return record

    def _decision_from(self, ctx: RunContext) -> Decision:
        """Extract the Decision the validate node stored in the context."""
        outputs = ctx.get(self.validate_node_id)
        decision = outputs.get("decision")
        if isinstance(decision, Decision):
            return decision
        if isinstance(decision, dict):
            return Decision.from_dict(decision)
        # No decision produced -> treat as stop to avoid an infinite loop.
        return Decision(action=DecisionAction.STOP.value, reason="no decision emitted")
