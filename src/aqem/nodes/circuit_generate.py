"""circuit_generate node — plan the circuit variants for the chosen strategy.

The actual variant arrays are built inside the executor (so the heavy numpy /
mitiq objects never need to live in the artifact store). This node records the
*plan* — how many ZNE scales x twirls x bases will run — which feeds the
``execute`` node's shot prediction and the audit trail.
"""

from __future__ import annotations

from ..braket_mitiq.observable_tools import pauli_grouping
from ..dag.node import Node
from ..dag.context import RunContext
from ..models import NodeResult, Strategy, Technique


class CircuitGenerateNode(Node):
    node_id = "circuit_generate"
    dependencies = ("strategy_select", "readout_calibrate")
    invalidates = ("execute", "post_process")

    def run(self, ctx: RunContext) -> NodeResult:
        strategy = Strategy.from_dict(ctx.get("strategy_select")["strategy"])
        bases, _ = pauli_grouping(ctx.problem.observable)

        n_scales = len(strategy.zne_scale_factors) if strategy.uses(Technique.ZNE.value) else 1
        n_twirls = max(1, strategy.twirl_count)
        n_variants = n_scales * n_twirls
        n_executables = n_variants * len(bases)

        plan = {
            "n_scales": n_scales,
            "n_twirls": n_twirls,
            "n_bases": len(bases),
            "n_variants": n_variants,
            "n_executables": n_executables,
            "uses_pt": strategy.uses(Technique.PT.value),
            "uses_rem": strategy.uses(Technique.REM.value),
            "uses_zne": strategy.uses(Technique.ZNE.value),
        }
        ctx.put(self.node_id, {"plan": plan})
        return NodeResult(node_id=self.node_id, outputs={"plan": plan})
