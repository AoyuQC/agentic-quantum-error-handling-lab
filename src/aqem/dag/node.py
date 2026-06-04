"""Node abstraction for the deterministic DAG state machine.

Each node declares its upstream ``dependencies`` and the set of nodes it
``invalidates`` when re-run (the cascade). ``run(ctx)`` performs the node's work
— any external effect must first pass ``ctx.policy.check`` — and returns a
:class:`NodeResult`, stashing its outputs in ``ctx.store`` for downstream nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import NodeResult
from .context import RunContext


class Node(ABC):
    """Base class for a DAG node.

    Subclasses set ``node_id``, ``dependencies`` and ``invalidates`` and
    implement :meth:`run`.
    """

    node_id: str = ""
    dependencies: tuple[str, ...] = ()
    # Nodes whose cached results become stale when THIS node re-runs.
    invalidates: tuple[str, ...] = ()

    @abstractmethod
    def run(self, ctx: RunContext) -> NodeResult:
        """Execute the node, returning a NodeResult and updating ctx.store."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Node {self.node_id} deps={self.dependencies}>"
