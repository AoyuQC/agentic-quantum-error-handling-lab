"""report node — package the final estimate, shot ledger, and audit trail.

The efficiency comparison against the static baseline is assembled by the CLI
(Phase L4), which runs both the baseline and this adaptive loop; this node emits
the adaptive side: the final estimate, total shots used, and the Policy audit.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..models import Estimate, NodeResult


class ReportNode(Node):
    node_id = "report"
    dependencies = ("validate",)
    invalidates = ()

    def run(self, ctx: RunContext) -> NodeResult:
        estimate = Estimate.from_dict(ctx.get("post_process")["estimate"])
        validate = ctx.get("validate")

        outputs = {
            "estimate": estimate.to_dict(),
            "shots_used": ctx.policy.budget.shots_used,
            "error_estimate": validate.get("error_estimate"),
            "audit": ctx.policy.audit.records,
            "decision": validate.get("decision"),
        }
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs)
