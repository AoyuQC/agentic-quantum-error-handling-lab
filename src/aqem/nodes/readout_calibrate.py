"""readout_calibrate node — build the REM inverse confusion matrix.

Skipped (no-op) when the strategy does not use REM. On RETRY_CALIBRATION it
re-runs with more readout twirls. The live numpy calibration object is stashed
under a private key so downstream ``execute`` can use it without round-tripping
through JSON.
"""

from __future__ import annotations

import numpy as np

from ..dag.node import Node
from ..dag.context import RunContext
from ..models import Calibration, NodeResult, Strategy, Technique
from ..policy import Action, ActionRequest
from ..tools.braket_tool import ReadoutCalibration, calibrate_readout


class ReadoutCalibrateNode(Node):
    node_id = "readout_calibrate"
    dependencies = ("strategy_select",)
    invalidates = ("circuit_generate", "execute", "post_process")

    def run(self, ctx: RunContext) -> NodeResult:
        strategy = Strategy.from_dict(ctx.get("strategy_select")["strategy"])
        if not strategy.uses(Technique.REM.value):
            outputs = {"calibration": None, "skipped": True}
            ctx.put(self.node_id, outputs)
            return NodeResult(node_id=self.node_id, status="skipped", outputs=outputs)

        nq = ctx.problem.num_qubits
        # More twirls on a calibration retry (adaptive escalation).
        retries = ctx.policy.retry_count(self.node_id)
        rem_twirls = strategy.rem_twirls * (2 ** retries)
        shots = strategy.shot_per_base * strategy.overhead

        req = ActionRequest(
            action=Action.RUN_READOUT_CONFUSION_MATRIX,
            node_id=self.node_id,
            params={"rem_twirls": rem_twirls},
            predicted_shots=shots,
        )
        decision = ctx.policy.check(req)
        if not decision.approved:
            return NodeResult(node_id=self.node_id, status="failed", error=decision.reason)

        cal: ReadoutCalibration = calibrate_readout(nq, ctx.device, rem_twirls, shots)
        ctx.policy.charge(shots=shots)

        # Serializable summary for audit/report...
        summary = Calibration(
            inverse_confusion_matrix=np.asarray(cal.inverse_confusion_matrix).tolist(),
            qubit_readout_errors=cal.qubit_readout_errors,
            quality=cal.quality,
            rem_twirls=rem_twirls,
            shots_used=shots,
        )
        # ...and the live object for the executor.
        outputs = {"calibration": summary.to_dict(), "_live": cal}
        ctx.put(self.node_id, outputs)
        return NodeResult(
            node_id=self.node_id,
            outputs={"calibration": summary.to_dict()},
            shots_used=shots,
        )
