"""Run context shared across DAG nodes.

Holds the immutable task spec, the live Policy/Budget, an artifact store for
node outputs (so downstream nodes and the invalidation cascade can read/clear
them), and run configuration (device, VLM client, shot knobs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import Problem
from ..policy import Policy


@dataclass
class RunContext:
    """Per-session state threaded through every node.

    Attributes:
        problem: the estimation task (observable, qubits, target accuracy).
        circuit: the runnable target circuit.
        device: Braket device / LocalSimulator to execute on.
        policy: the deterministic Policy gating all side effects.
        config: free-form run configuration (probe shots, baseline knobs, ...).
        vlm: optional VLM client (None until Phase L3 / when offline).
        store: artifact store — node_id -> outputs dict; cleared on invalidation.
    """

    problem: Problem
    circuit: Any
    device: Any
    policy: Policy
    config: dict[str, Any] = field(default_factory=dict)
    vlm: Optional[Any] = None
    store: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- artifact helpers ---------------------------------------------------
    def put(self, node_id: str, outputs: dict[str, Any]) -> None:
        self.store[node_id] = outputs

    def get(self, node_id: str) -> dict[str, Any]:
        return self.store.get(node_id, {})

    def has(self, node_id: str) -> bool:
        return node_id in self.store

    def clear(self, node_ids: list[str]) -> None:
        for nid in node_ids:
            self.store.pop(nid, None)
