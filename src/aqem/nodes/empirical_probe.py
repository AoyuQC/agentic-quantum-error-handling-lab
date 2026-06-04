"""empirical_probe node — cheap diagnostic characterization (no device metadata).

Runs the readout + GHZ probes, builds histogram figures, and classifies the
dominant error source. In Phase L2 classification is the deterministic
``classify_probes`` heuristic; Phase L3 adds a VLM verdict on the histograms.
"""

from __future__ import annotations

from ..dag.node import Node
from ..dag.context import RunContext
from ..models import NodeResult
from ..policy import Action, ActionRequest
from ..probes.circuits import probe_circuits
from ..probes.classify import classify_probes
from ..probes.histograms import histogram_figure
from ..tools.braket_tool import run_probe
from ..tools.vlm_tool import classify_probe_with_vlm


class EmpiricalProbeNode(Node):
    node_id = "empirical_probe"
    dependencies = ()
    invalidates = ()

    def run(self, ctx: RunContext) -> NodeResult:
        nq = ctx.problem.num_qubits
        shots = int(ctx.config.get("probe_shots", 2000))
        probes = probe_circuits(nq)

        # Probing is a side effect: gate it on the budget.
        predicted = shots * len(probes)
        req = ActionRequest(
            action=Action.RUN_READOUT_CONFUSION_MATRIX,  # probing precedes calibration
            node_id=self.node_id,
            params={"kind": "empirical_probe"},
            predicted_shots=predicted,
        )
        decision = ctx.policy.check(req)
        if not decision.approved:
            return NodeResult(node_id=self.node_id, status="failed", error=decision.reason)

        counts = {name: run_probe(circ, ctx.device, shots) for name, circ in probes.items()}
        ctx.policy.charge(shots=predicted)

        # Deterministic classification always runs (the rules floor / fallback).
        classification = classify_probes(counts["readout"], counts["ghz"], nq)
        merged = classification.to_dict()

        zeros, ones = "0" * nq, "1" * nq
        plots = [
            {
                "name": "readout_probe",
                "format": "plotly",
                "data": histogram_figure(counts["readout"], "readout probe (prep |0...0>)", [zeros]),
            },
            {
                "name": "ghz_probe",
                "format": "plotly",
                "data": histogram_figure(counts["ghz"], "GHZ probe", [zeros, ones]),
            },
        ]

        # VLM augmentation: a confident verdict overrides the dominant-error
        # class and may add (never drop) a suggested focus technique.
        vlm_classification = None
        if ctx.vlm is not None:
            threshold = float(ctx.config.get("vlm_confidence_threshold", 0.5))
            vlm_classification = classify_probe_with_vlm(ctx.vlm, plots, threshold)
            if not vlm_classification.get("degraded"):
                merged["dominant_error"] = vlm_classification["dominant_error"]
                for t in vlm_classification.get("suggested_focus", []):
                    if t not in merged["suggested_focus"]:
                        merged["suggested_focus"].append(t)
                merged["source"] = "vlm+rules"

        outputs = {
            "counts": counts,
            "classification": merged,
            "rules_classification": classification.to_dict(),
            "vlm_classification": vlm_classification,
        }
        ctx.put(self.node_id, outputs)
        return NodeResult(
            node_id=self.node_id, outputs=outputs, plots=plots, shots_used=predicted
        )
