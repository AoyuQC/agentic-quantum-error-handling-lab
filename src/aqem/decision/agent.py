"""LLM orchestration agent — the loop's decider (rules are the fallback).

The design diagram (``docs/quantum-calibration-agent.drawio``) splits the loop
into a VLM *tool* (image analysis only) and an *orchestrator* that owns the
control flow. This module is that orchestrator: given the target, the running
history of (strategy tried -> error achieved -> VLM's read of the plot), the
remaining budget and the iteration count, it chooses the **next parameter set**
to try and decides whether to continue or STOP.

Unlike the VLM, the orchestrator legitimately sees the true error and the
target — it is the decider, not a tool — so it can issue a reasoned STOP when
the target is met *or* when escalations have plateaued and the target looks
infeasible given the remaining budget. The VLM, by contrast, only ever sees the
ZNE plot and must never end the loop.

The agent reuses the same Bedrock Claude client as the VLM (text-only call). On
any failure (no client, non-JSON output, schema mismatch, low confidence) it
returns ``None`` so the deterministic rules in :mod:`aqem.decision.rules` take
over — keeping the loop runnable offline and in CI, and preserving the audit
story.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from ..baseline.full_stack import FACTORY_ESCALATION  # noqa: F401  (doc reference)
from ..models import Decision, DecisionAction, Strategy, Technique

logger = logging.getLogger(__name__)

# The full parameter space the orchestrator may choose from.
_TECHNIQUES = (Technique.REM.value, Technique.PT.value, Technique.ZNE.value)
_FACTORIES = ("Linear", "Exp", "Richardson", "Poly")
_ACTIONS = tuple(a.value for a in DecisionAction)


class OrchestratorDecision(BaseModel):
    """Structured output of the orchestration agent.

    ``action`` controls the loop (STOP or a RETRY_* mode). The remaining fields
    describe the *next* strategy to try on a retry; they are ignored on STOP.
    """

    action: Literal["stop", "retry_shots", "retry_calibration", "retry_strategy"] = Field(
        description="STOP, or which retry mode to drive next."
    )
    reason: str = Field(default="", description="Short justification for the decision.")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this decision."
    )
    techniques: list[Literal["REM", "PT", "ZNE"]] = Field(
        default_factory=list,
        description="Mitigation techniques to apply next (must include REM).",
    )
    zne_factory: Literal["Linear", "Exp", "Richardson", "Poly"] = Field(
        default="Exp", description="ZNE extrapolation factory to use next."
    )
    zne_scale_factors: list[int] = Field(
        default_factory=lambda: [1, 3],
        description="Noise scale factors for ZNE folding (ascending, start at 1).",
    )
    twirl_count: int = Field(default=8, ge=1, le=64, description="Pauli-twirl variants.")
    shot_per_base: int = Field(
        default=4000, ge=100, le=200_000, description="Shots per measurement basis."
    )
    overhead: int = Field(default=3, ge=1, le=16, description="REM quasi-prob shot multiplier.")


_SCHEMA_HINT = (
    '{"action": "stop|retry_shots|retry_calibration|retry_strategy", '
    '"reason": "string", "confidence": 0.0-1.0, '
    '"techniques": ["REM"|"PT"|"ZNE", ...], '
    '"zne_factory": "Linear|Exp|Richardson|Poly", '
    '"zne_scale_factors": [1, 3, ...], "twirl_count": int, '
    '"shot_per_base": int, "overhead": int}'
)


def _system_prompt() -> str:
    return (
        "You are the orchestration agent for an adaptive quantum error-mitigation "
        "(QEM) loop. Each iteration runs a chosen mitigation strategy on a noisy "
        "circuit and measures the absolute error |estimate - ideal| against a "
        "target accuracy. YOU decide what to try next and when to stop; an "
        "image-analysis VLM only describes the extrapolation plot and never "
        "decides.\n\n"
        "The knobs you control (the next Strategy):\n"
        "- techniques: ordered subset of {REM, PT, ZNE}. REM (readout-error "
        "mitigation) is the cheap floor and should almost always be included. "
        "Add ZNE (zero-noise extrapolation) for gate/coherent noise; add PT "
        "(Pauli twirling) to tame coherent errors so ZNE extrapolates cleanly.\n"
        "- zne_factory: Linear < Exp < Richardson grow in flexibility (and "
        "variance); Poly(2) is an alternative. Richardson needs >=3 scale points.\n"
        "- zne_scale_factors: ascending odd integers starting at 1 (e.g. [1,3], "
        "[1,3,5]). More points => better extrapolation but more shots.\n"
        "- twirl_count, shot_per_base, overhead: raising these lowers variance "
        "(shrinks the error bar) at linear shot cost.\n\n"
        "Decision policy:\n"
        "1. If the latest error <= target: action=stop (target met).\n"
        "2. If the error is a small multiple of target and the bar is the limiter: "
        "action=retry_shots and raise shot_per_base/twirl_count.\n"
        "3. If the VLM reports a non-monotone extrapolation or outliers: "
        "action=retry_strategy and change zne_factory or add scale points.\n"
        "4. If the VLM reports a readout anomaly: action=retry_calibration.\n"
        "5. If far from target: action=retry_strategy and escalate techniques "
        "(add ZNE, then PT) before touching the factory.\n"
        "6. STOP EARLY if escalations have plateaued — error barely improves "
        "across the last attempts AND the remaining budget cannot plausibly close "
        "the gap (the target is infeasible on this noise model). Say so in reason. "
        "Do NOT keep burning the budget on a hopeless target.\n\n"
        "Be decisive and frugal: prefer the cheapest change that can move the "
        "error, escalate only when needed, and stop the moment further effort is "
        "either unnecessary (target met) or futile (plateaued + infeasible)."
    )


def _user_prompt(
    *,
    target: float,
    current_error: Optional[float],
    current_error_bar: float,
    current_strategy: Strategy,
    attempts: list[dict[str, Any]],
    vlm_analysis: Optional[dict[str, Any]],
    remaining_shots: Optional[int],
    iteration: int,
    max_iterations: int,
) -> str:
    lines: list[str] = []
    lines.append(f"target_accuracy: {target}")
    metric = current_error if current_error is not None else current_error_bar
    lines.append(
        f"latest_error: {metric:.5f}"
        + ("" if current_error is not None else " (error bar; no exact reference)")
    )
    lines.append(f"latest_error_bar: {current_error_bar:.5f}")
    lines.append(f"iteration: {iteration} of max {max_iterations}")
    lines.append(
        "remaining_shots: "
        + ("unbounded" if remaining_shots is None else str(remaining_shots))
    )
    lines.append("current_strategy: " + json.dumps(current_strategy.to_dict()))

    if attempts:
        lines.append("\nattempt_history (oldest first):")
        for a in attempts:
            err = a.get("error")
            err_s = f"{err:.5f}" if isinstance(err, (int, float)) else "n/a"
            lines.append(
                f"  - iter {a.get('iteration')}: techniques={a.get('techniques')}, "
                f"factory={a.get('zne_factory')}, scales={a.get('zne_scale_factors')}, "
                f"error={err_s}, error_bar={a.get('error_bar')}, "
                f"shots_used={a.get('shots_used')}"
            )

    if vlm_analysis:
        # Pass only the analytic fields — never the VLM's (now-ignored) action.
        keep = {
            k: vlm_analysis.get(k)
            for k in (
                "extrapolation_monotone",
                "has_outliers",
                "readout_anomaly",
                "improvement_meaningful",
                "rationale",
                "confidence",
            )
            if k in vlm_analysis
        }
        lines.append("\nvlm_plot_analysis: " + json.dumps(keep))
    else:
        lines.append("\nvlm_plot_analysis: none (no plot or VLM degraded)")

    lines.append(
        "\nDecide the next action and the next Strategy. Respond with EXACTLY ONE "
        "JSON object and nothing else, matching this schema:\n" + _SCHEMA_HINT
    )
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a possibly chatty response."""
    import re

    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _sanitize_strategy(d: OrchestratorDecision, base: Strategy) -> Strategy:
    """Clamp the agent's proposed knobs to the valid, in-bounds parameter set."""
    techniques = [t for t in d.techniques if t in _TECHNIQUES]
    # REM is the floor; never let the agent drop it.
    if Technique.REM.value not in techniques:
        techniques = [Technique.REM.value] + techniques
    # De-dup while preserving order.
    seen: set[str] = set()
    techniques = [t for t in techniques if not (t in seen or seen.add(t))]

    factory = d.zne_factory if d.zne_factory in _FACTORIES else base.zne_factory

    scales = sorted({int(s) for s in d.zne_scale_factors if int(s) >= 1}) or [1, 3]
    if scales[0] != 1:
        scales = [1] + scales
    # Richardson needs >= 3 points; pad if the agent under-specified.
    if factory == "Richardson" and len(scales) < 3:
        scales = sorted(set(scales) | {1, 3, 5})

    return Strategy(
        techniques=techniques,
        zne_scale_factors=scales,
        zne_factory=factory,
        twirl_count=int(d.twirl_count),
        rem_twirls=base.rem_twirls,
        shot_per_base=int(d.shot_per_base),
        overhead=int(d.overhead),
    )


def propose_decision(
    vlm: Optional[Any],
    *,
    target: float,
    current_error: Optional[float],
    current_error_bar: float,
    current_strategy: Strategy,
    attempts: list[dict[str, Any]],
    vlm_analysis: Optional[dict[str, Any]],
    remaining_shots: Optional[int],
    iteration: int,
    max_iterations: int,
    confidence_threshold: float = 0.5,
) -> Optional[tuple[Decision, Optional[Strategy]]]:
    """Ask the orchestration agent for the next decision + strategy.

    Returns:
        ``(Decision, next_strategy_or_None)`` on success. ``next_strategy`` is
        None for a STOP. Returns ``None`` to signal the caller should fall back
        to the deterministic rules (no client, error, bad output, or the agent's
        confidence is below ``confidence_threshold``).
    """
    if vlm is None:
        return None

    from ..tools.vlm_tool import _run_async  # reuse the sync<->async bridge

    prompt = _system_prompt() + "\n\n" + _user_prompt(
        target=target,
        current_error=current_error,
        current_error_bar=current_error_bar,
        current_strategy=current_strategy,
        attempts=attempts,
        vlm_analysis=vlm_analysis,
        remaining_shots=remaining_shots,
        iteration=iteration,
        max_iterations=max_iterations,
    )

    try:
        # Text-only reasoning call (no images) on the same Bedrock client.
        raw = _run_async(vlm.analyze_images(prompt, []))
    except Exception as e:
        logger.error("orchestrator agent call failed: %s", e)
        return None

    parsed = _extract_json(raw if isinstance(raw, str) else str(raw))
    if parsed is None:
        logger.warning("orchestrator agent returned non-JSON output; using rules")
        return None
    try:
        out = OrchestratorDecision.model_validate(parsed)
    except ValidationError as e:
        logger.warning("orchestrator agent output failed schema validation: %s", e)
        return None

    if out.confidence < confidence_threshold:
        logger.info("orchestrator agent confidence below threshold; using rules")
        return None

    if out.action == DecisionAction.STOP.value:
        decision = Decision(
            action=DecisionAction.STOP.value,
            reason=f"agent: {out.reason}" if out.reason else "agent: stop",
            source="agent",
        )
        return decision, None

    if out.action not in _ACTIONS:
        return None

    next_strategy = _sanitize_strategy(out, current_strategy)
    decision = Decision(
        action=out.action,
        reason=f"agent: {out.reason}" if out.reason else f"agent: {out.action}",
        source="agent",
    )
    return decision, next_strategy
