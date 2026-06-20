"""Run context shared across DAG nodes.

Holds the immutable task spec, the live Policy/Budget, an artifact store for
node outputs (so downstream nodes and the invalidation cascade can read/clear
them), and run configuration (device, VLM client, shot knobs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
        tools: the tool transport (in-process by default, MCP/Gateway when opted
            in) every node calls for probe/calibrate/mitigate/VLM side effects.
        store: artifact store — node_id -> outputs dict; cleared on invalidation.
    """

    problem: Problem
    circuit: Any
    device: Any
    policy: Policy
    config: dict[str, Any] = field(default_factory=dict)
    vlm: Optional[Any] = None
    tools: Optional[Any] = None
    store: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Live progress sink + current iteration, set by the engine each pass so a
    # node can stream sub-events (e.g. LLM tokens) while it runs. ``emit`` is a
    # no-op when no observer is wired (offline / tests).
    emit: Callable[[dict[str, Any]], None] = lambda ev: None
    iteration: int = 0

    # -- tool transport -----------------------------------------------------
    def tool_client(self):
        """The tool transport, defaulting to in-process if none was wired in.

        Lazily builds (and caches) an ``InProcessToolClient`` from ``device`` +
        ``vlm`` so call sites and tests that construct a context without an
        explicit ``tools`` keep working with identical (in-process) numerics.
        """
        if self.tools is None:
            from ..tools.client import InProcessToolClient

            self.tools = InProcessToolClient(device=self.device, vlm=self.vlm)
        return self.tools

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
