"""Assemble and run the adaptive QEM loop.

Wires a :class:`~aqem.dag.context.RunContext` (problem, device, Policy) to the
default node pipeline and the :class:`~aqem.dag.engine.DAGEngine`, returning the
engine's :class:`~aqem.dag.engine.RunRecord`. This is the single entry point the
L4 CLI and the integration tests call.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from braket.circuits import Circuit
from braket.devices import Device

from .dag import DAGEngine, RunContext, RunRecord
from .models import Budget, Problem
from .nodes import default_nodes
from .policy import AuditLog, Policy


def run_adaptive_loop(
    problem: Problem,
    circuit: Circuit,
    device: Device,
    budget: Budget,
    config: Optional[dict[str, Any]] = None,
    vlm: Any = None,
    tools: Any = None,
    audit_path: Optional[str] = None,
    max_iterations: int = 8,
    max_retries_per_node: int = 3,
    seed: Optional[int] = None,
    observer: Optional[Any] = None,
) -> RunRecord:
    """Run the full adaptive loop and return the engine RunRecord.

    Args:
        problem: the estimation task.
        circuit: the runnable target circuit.
        device: a Braket device / LocalSimulator with a noise model.
        budget: the shot/cost ledger (Policy hard gate).
        config: run knobs (probe_shots, shot_per_base, overhead, rem_twirls,
            use_ideal_for_validation, ...).
        vlm: optional VLM client (Phase L3); None runs rules-only.
        tools: optional tool transport (``ToolClient``); defaults to the
            in-process client bound to ``device``/``vlm``. Pass an
            ``McpToolClient`` to route tool calls through the AgentCore Gateway.
        audit_path: optional JSONL path for the Policy audit log.
        max_iterations: safety bound on adaptive retries.
        max_retries_per_node: per-node retry cap enforced by Policy.
        seed: if given, seed Python's ``random`` so the (twirl-based) numerics
            are reproducible. The vendored Mitiq twirling uses the global RNG.

    Returns:
        RunRecord with status ("stopped"/"exhausted"/"failed"), the final
        Decision, per-node results, and final outputs (the report node).
    """
    if seed is not None:
        random.seed(seed)
        # The Braket density-matrix simulator samples shots via numpy's global
        # RNG, and mitiq twirling uses Python's `random`; seed both so a given
        # seed yields a reproducible run regardless of prior RNG consumption.
        import numpy as np

        np.random.seed(seed)

    policy = Policy(
        budget=budget,
        audit=AuditLog(audit_path),
        max_retries_per_node=max_retries_per_node,
    )
    if tools is None:
        from .tools.client import InProcessToolClient

        tools = InProcessToolClient(device=device, vlm=vlm)
    ctx = RunContext(
        problem=problem,
        circuit=circuit,
        device=device,
        policy=policy,
        config=dict(config or {}),
        vlm=vlm,
        tools=tools,
    )
    engine = DAGEngine(default_nodes(), max_iterations=max_iterations, observer=observer)
    return engine.run(ctx)
