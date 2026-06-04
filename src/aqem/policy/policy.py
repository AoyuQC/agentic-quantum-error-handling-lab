"""Deterministic Policy: the gate between "agent proposes" and "tool executes".

Every controlled action passes ``Policy.check`` before it runs. The checks, in
order (design-doc §6.3 / §7 / §2.2):

  1. action-set gate    — reject anything not in the controlled :class:`Action` set
  2. budget hard gate   — reject if the prediction would overrun the shot/cost budget
  3. no-recalibration   — reject any attempt at vendor device recalibration / metadata
  4. retry-cap gate     — reject if a node has exceeded its retry budget

Every call is audited, approved or not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Budget
from .actions import Action, ActionRequest
from .audit import AuditLog

# Substrings that signal a (forbidden) attempt to query/recalibrate device
# internals. The agent performs *mitigation diagnosis*, never QPU recalibration.
_RECALIBRATION_DENYLIST = (
    "recalibrat",
    "calibrate_device",
    "device_properties",
    "vendor_calibration",
    "qubit_frequency",
    "pulse",
    "t1",
    "t2",
)


@dataclass
class PolicyDecision:
    """The verdict for a single :class:`ActionRequest`."""

    approved: bool
    reason: str

    def __bool__(self) -> bool:  # allow `if policy.check(req): ...`
        return self.approved


class Policy:
    """Stateful deterministic policy guarding a single agent session."""

    def __init__(
        self,
        budget: Budget,
        audit: AuditLog | None = None,
        max_retries_per_node: int = 3,
    ):
        self.budget = budget
        # NB: AuditLog defines __len__, so an empty log is falsy — must test
        # identity, not truthiness, or a passed-but-empty log gets discarded.
        self.audit = audit if audit is not None else AuditLog()
        self.max_retries_per_node = max_retries_per_node
        self._retry_counts: dict[str, int] = {}

    # -- retry bookkeeping --------------------------------------------------
    def record_retry(self, node_id: str) -> int:
        """Increment and return the retry count for a node."""
        self._retry_counts[node_id] = self._retry_counts.get(node_id, 0) + 1
        return self._retry_counts[node_id]

    def retry_count(self, node_id: str) -> int:
        return self._retry_counts.get(node_id, 0)

    # -- the gate -----------------------------------------------------------
    def check(self, request: ActionRequest) -> PolicyDecision:
        """Evaluate a request against all gates, audit, and return the verdict."""
        approved, reason = self._evaluate(request)
        self.audit.append(
            {
                "node_id": request.node_id,
                "action": request.action.value
                if isinstance(request.action, Action)
                else request.action,
                "params": request.params,
                "predicted_shots": request.predicted_shots,
                "predicted_cost": request.predicted_cost,
                "approved": approved,
                "reason": reason,
                "budget_remaining_shots": self.budget.remaining_shots(),
            }
        )
        return PolicyDecision(approved=approved, reason=reason)

    def _evaluate(self, request: ActionRequest) -> tuple[bool, str]:
        # 1. action-set gate
        if not isinstance(request.action, Action):
            try:
                Action(request.action)
            except ValueError:
                return False, f"action '{request.action}' is not in the controlled action set"

        # 3. no-device-recalibration guard (checked before spending anything)
        haystack = " ".join(
            [str(request.action), *[str(k) for k in request.params], *[str(v) for v in request.params.values()]]
        ).lower()
        for term in _RECALIBRATION_DENYLIST:
            if term in haystack:
                return False, f"device recalibration / metadata access is forbidden (matched '{term}')"

        # 4. retry-cap gate (retries are recorded by the engine via record_retry)
        if request.node_id and self.retry_count(request.node_id) > self.max_retries_per_node:
            return False, (
                f"node '{request.node_id}' exceeded retry cap "
                f"({self.max_retries_per_node})"
            )

        # 2. budget hard gate
        if self.budget.would_exceed(request.predicted_shots, request.predicted_cost):
            return False, (
                f"budget exceeded: predicted {request.predicted_shots} shots / "
                f"{request.predicted_cost} cost vs remaining "
                f"{self.budget.remaining_shots()} shots / {self.budget.remaining_cost()} cost"
            )

        return True, "approved"

    def charge(self, shots: int = 0, cost: float = 0.0) -> None:
        """Record consumption against the budget after an approved action runs."""
        self.budget.charge(shots=shots, cost=cost)
