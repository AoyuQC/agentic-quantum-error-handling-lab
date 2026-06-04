"""Core data models for the AQEM agent.

Dataclasses follow the ``to_dict``/``from_dict`` style of the NVIDIA blueprint
so results stay JSON-serializable (for audit logs, reports, and a future
subprocess/Gateway boundary). Only the models needed through Phase L1 are
defined here; the DAG/decision models (Strategy, Calibration, Variants,
Decision, NodeResult) are added in Phase L2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class Problem:
    """The estimation task the agent is asked to solve.

    A Hamiltonian / observable is expressed as a list of weighted Pauli terms,
    ``[(coeff, pauli_string), ...]`` where each ``pauli_string`` has one
    character per qubit drawn from {I, X, Y, Z} — the same convention used by
    the vendored ``observable_tools.pauli_grouping``.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        observable: Weighted Pauli decomposition of the observable to estimate.
        target_accuracy: Desired absolute error |estimate - ideal|; the loop
            early-stops once the error bar is within this.
        description: Human-readable label for reports.
    """

    num_qubits: int
    observable: list[tuple[float, str]]
    target_accuracy: float = 0.05
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "observable": [[float(c), p] for c, p in self.observable],
            "target_accuracy": self.target_accuracy,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Problem":
        return cls(
            num_qubits=data["num_qubits"],
            observable=[(float(c), p) for c, p in data["observable"]],
            target_accuracy=data.get("target_accuracy", 0.05),
            description=data.get("description", ""),
        )


@dataclass
class Budget:
    """Shot / cost ledger. The Policy cost gate refuses actions that would
    exceed the remaining budget — the central efficiency mechanism.

    A ``None`` ceiling means "no limit" (e.g. no cost ceiling on a local
    simulator).
    """

    shots_total: Optional[int] = None
    cost_total: Optional[float] = None
    shots_used: int = 0
    cost_used: float = 0.0

    def remaining_shots(self) -> Optional[int]:
        if self.shots_total is None:
            return None
        return self.shots_total - self.shots_used

    def remaining_cost(self) -> Optional[float]:
        if self.cost_total is None:
            return None
        return self.cost_total - self.cost_used

    def would_exceed(self, predicted_shots: int = 0, predicted_cost: float = 0.0) -> bool:
        """True if charging the prediction would overrun either ceiling."""
        if self.shots_total is not None and self.shots_used + predicted_shots > self.shots_total:
            return True
        if self.cost_total is not None and self.cost_used + predicted_cost > self.cost_total:
            return True
        return False

    def charge(self, shots: int = 0, cost: float = 0.0) -> None:
        """Record consumption against the ledger (call only after a run)."""
        self.shots_used += shots
        self.cost_used += cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Budget":
        return cls(**data)


@dataclass
class Estimate:
    """An observable estimate with its uncertainty and provenance.

    Attributes:
        value: The (mitigated or raw) expectation value.
        error_bar: Estimated 1-sigma uncertainty (e.g. jackknife std over twirls).
        shots_used: Total shots spent producing this estimate.
        techniques: Which mitigation techniques were applied (e.g. ["REM", "ZNE"]).
        zne_data: Optional {scale_factor: expectation} used for the extrapolation.
        plots: List of {name, format, data} plot records (e.g. plotly JSON).
        metadata: Free-form extras (factory used, twirl counts, ...).
    """

    value: float
    error_bar: float = 0.0
    shots_used: int = 0
    techniques: list[str] = field(default_factory=list)
    zne_data: dict[str, float] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Estimate":
        return cls(**data)


class Technique(str, Enum):
    """Mitiq mitigation techniques the strategy may select."""

    REM = "REM"   # readout error mitigation
    PT = "PT"     # Pauli twirling
    ZNE = "ZNE"   # zero-noise extrapolation


@dataclass
class Strategy:
    """The mitigation recipe chosen by ``strategy_select``.

    Starts minimal (often REM-only) and is escalated only when ``validate``
    shows the target is unmet — the minimum-sufficient-mitigation bias.

    Attributes:
        techniques: ordered subset of {REM, PT, ZNE} to apply.
        zne_scale_factors: noise scale factors for ZNE folding.
        zne_factory: extrapolation factory name (Linear/Exp/Richardson/Poly).
        twirl_count: number of Pauli-twirl variants averaged.
        rem_twirls: number of readout-twirl samples for REM calibration.
        shot_per_base: shots per measurement basis (before overhead).
        overhead: shot multiplier (REM quasi-probability overhead).
    """

    techniques: list[str] = field(default_factory=lambda: [Technique.REM.value])
    zne_scale_factors: list[int] = field(default_factory=lambda: [1, 3])
    zne_factory: str = "Exp"
    twirl_count: int = 8
    rem_twirls: int = 20
    shot_per_base: int = 4000
    overhead: int = 3

    def uses(self, technique: str) -> bool:
        return technique in self.techniques

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strategy":
        return cls(**data)


@dataclass
class Calibration:
    """REM readout calibration output (inverse confusion matrix + quality).

    The inverse confusion matrix is kept as a nested list so the object stays
    JSON-serializable; convert back to ``np.array`` at use sites.
    """

    inverse_confusion_matrix: list[list[float]] = field(default_factory=list)
    qubit_readout_errors: list[float] = field(default_factory=list)
    quality: float = 0.0
    rem_twirls: int = 0
    shots_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        return cls(**data)


class DecisionAction(str, Enum):
    """Outcomes of the ``validate`` step, mapped to controlled Policy actions."""

    STOP = "stop"
    RETRY_SHOTS = "retry_shots"
    RETRY_CALIBRATION = "retry_calibration"
    RETRY_STRATEGY = "retry_strategy"


@dataclass
class Decision:
    """The validate step's verdict on whether/how to continue.

    Attributes:
        action: STOP or one of the RETRY_* modes.
        reason: human-readable justification (rules + optional VLM rationale).
        invalidate: node ids whose cached results must be recomputed on retry.
        source: "rules" or "vlm+rules" — provenance of the decision.
    """

    action: str
    reason: str = ""
    invalidate: list[str] = field(default_factory=list)
    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        return cls(**data)


@dataclass
class NodeResult:
    """Result of executing one DAG node.

    Mirrors the ``{status, results, ...}`` contract of the NVIDIA blueprint's
    runner so node functions stay subprocess/Gateway-portable later.
    """

    node_id: str
    status: str = "success"            # "success" | "failed" | "skipped"
    outputs: dict[str, Any] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    shots_used: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeResult":
        return cls(**data)
