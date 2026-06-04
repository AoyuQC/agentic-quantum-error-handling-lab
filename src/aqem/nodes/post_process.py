"""post_process node — assemble the Estimate and the ZNE extrapolation plot.

The numeric pipeline (REM correction, twirl averaging, ZNE extrapolation) runs
inside ``execute``/``run_mitigation``; this node packages those results into an
:class:`Estimate`, totals the shots spent so far this iteration, and renders the
extrapolation figure the ``validate`` step (and the VLM, in L3) inspects.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..models import Estimate, NodeResult
from ..problems import ideal_expectation
from ..reporting.plots import zne_extrapolation_figure


class PostProcessNode(Node):
    node_id = "post_process"
    dependencies = ("execute",)
    invalidates = ("validate",)

    def run(self, ctx: RunContext) -> NodeResult:
        ex = ctx.get("execute")
        cal = ctx.get("readout_calibrate")
        cal_shots = (cal.get("calibration") or {}).get("shots_used", 0) if cal else 0

        estimate = Estimate(
            value=ex["value"],
            error_bar=ex["error_bar"],
            shots_used=ex["shots_used"] + cal_shots,
            techniques=ex["techniques"],
            zne_data=ex["zne_data"],
        )

        plots = []
        if len(estimate.zne_data) >= 2:
            # Reference line only if we cheaply know the ideal (local benchmarking).
            ideal = None
            if ctx.config.get("show_ideal_reference"):
                ideal = ideal_expectation(ctx.circuit, ctx.problem.observable)
            plots.append({
                "name": "zne_extrapolation",
                "format": "plotly",
                "data": zne_extrapolation_figure(
                    estimate.zne_data, estimate.value, ideal=ideal
                ),
            })

        outputs = {"estimate": estimate.to_dict()}
        ctx.put(self.node_id, outputs)
        return NodeResult(node_id=self.node_id, outputs=outputs, plots=plots)
