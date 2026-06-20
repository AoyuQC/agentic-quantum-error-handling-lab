"""Helper to stream LLM tokens out of a node as live progress sub-events.

Nodes that make an LLM call (the VLM analysis in ``empirical_probe`` / ``validate``
and the orchestration agent in ``validate``) build a token sink with
:func:`token_emitter`. Each text delta is pushed through ``ctx.emit`` as an
``llm_delta`` event tagged with the node, iteration, and which model spoke
(``"vlm"`` or ``"agent"``) — so the web UI can render the model's answer as it
streams, while the step is still running. When no observer is wired (offline /
tests) ``ctx.emit`` is a no-op and the sink simply does nothing useful.
"""

from __future__ import annotations

from typing import Callable

from ..dag.context import RunContext


def token_emitter(ctx: RunContext, node_id: str, role: str) -> Callable[[str], None]:
    """Return an ``on_token(delta)`` sink that emits ``llm_delta`` events."""

    def on_token(delta: str) -> None:
        if not delta:
            return
        ctx.emit({
            "event": "llm_delta",
            "node": node_id,
            "iteration": ctx.iteration,
            "role": role,
            "delta": delta,
        })

    return on_token
