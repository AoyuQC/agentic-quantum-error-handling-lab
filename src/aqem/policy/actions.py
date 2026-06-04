"""The controlled action set the agent may request.

Policy rejects anything outside this enum (design-doc §6.3). Each side-effecting
step in the DAG expresses its intent as an :class:`ActionRequest` carrying a
shot/cost prediction, which the cost gate checks against the remaining budget
before the action is allowed to run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    """Exactly the actions allowed by the design's controlled action set."""

    INCREASE_SHOTS = "increase_shots"
    RUN_READOUT_CONFUSION_MATRIX = "run_readout_confusion_matrix"
    RUN_READOUT_MITIGATION = "run_readout_mitigation"
    RUN_ZNE_SWEEP = "run_zne_sweep"
    RUN_PAULI_TWIRLING = "run_pauli_twirling"
    CHANGE_ZNE_FACTORY = "change_zne_factory"
    REDUCE_TECHNIQUE_SET = "reduce_technique_set"
    STOP_AND_REPORT = "stop_and_report"


@dataclass
class ActionRequest:
    """A request to perform one controlled action, with a cost prediction.

    Attributes:
        action: the requested :class:`Action`.
        node_id: the DAG node making the request (for audit).
        params: action-specific parameters.
        predicted_shots: shots this action is expected to consume.
        predicted_cost: monetary cost this action is expected to incur.
    """

    action: Action
    node_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    predicted_shots: int = 0
    predicted_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value if isinstance(self.action, Action) else self.action
        return d
