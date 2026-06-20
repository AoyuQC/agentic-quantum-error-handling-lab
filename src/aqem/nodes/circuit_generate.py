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
            "scale_factors": list(strategy.zne_scale_factors) if strategy.uses(Technique.ZNE.value) else [1],
        }
        outputs = {"plan": plan, **_folded_views(ctx.circuit, strategy)}
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs)


def _folded_views(circuit, strategy: Strategy) -> dict:
    """ASCII diagrams of the ZNE noise-folded circuits the mitigation will run.

    This is what actually differs between iterations: the *base* logical circuit
    is identical every pass, but ZNE re-runs it at increasing noise scale
    factors (gate-folded copies at depth ~x1, x3, x5, ...). Surfacing the folded
    set makes the per-iteration escalation visible (it was invisible when only
    the base circuit was shown). REM/PT change the measurement/averaging, not
    the drawn gate sequence, so we annotate them rather than redraw.
    """
    folds: list[dict] = []
    if strategy.uses(Technique.ZNE.value) and len(strategy.zne_scale_factors) >= 1:
        try:
            from mitiq.zne import construct_circuits

            folded = construct_circuits(circuit, scale_factors=list(strategy.zne_scale_factors))
            for sf, fc in zip(strategy.zne_scale_factors, folded):
                folds.append({
                    "scale": int(sf),
                    "depth": int(getattr(fc, "depth", 0)),
                    "n_gates": len(getattr(fc, "instructions", [])),
                    "diagram": _diagram(fc),
                })
        except Exception:  # folding/preview must never break the run
            folds = []
    if not folds:
        # No ZNE: the single circuit run as-is (scale 1).
        folds = [{
            "scale": 1,
            "depth": int(getattr(circuit, "depth", 0)),
            "n_gates": len(getattr(circuit, "instructions", [])),
            "diagram": _diagram(circuit),
        }]
    return {"folded_circuits": folds}


def _diagram(circuit) -> str:
    try:
        return str(circuit.diagram())
    except Exception:
        try:
            return str(circuit)
        except Exception:
            return ""
