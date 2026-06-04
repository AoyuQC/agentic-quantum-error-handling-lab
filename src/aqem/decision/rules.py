"""Deterministic, rules-first decision logic.

Two decision points in the loop:

  - ``select_strategy``  : probe classification + target + budget -> initial Strategy
  - ``decide``           : estimate (+ optional VLM verdict) -> Decision (STOP / RETRY_*)

Rules run first and are the sole fallback when no VLM verdict is supplied or its
confidence is below threshold (the VLM is integrated in Phase L3). The bias is
always **minimum-sufficient mitigation**: start with the cheapest technique the
probe implies (usually REM), escalate only on a failed validate, and stop the
instant the target is met.
"""

from __future__ import annotations

from typing import Any, Optional

from ..baseline.full_stack import FACTORY_ESCALATION
from ..models import Decision, DecisionAction, Strategy, Technique

# How close to target counts as "just needs more shots" rather than a
# structural problem with the mitigation.
_CLOSE_TO_TARGET_FACTOR = 2.0

# Dominant-error classes the probe step may report.
DOMINANT_READOUT = "readout"
DOMINANT_GATE_COHERENT = "gate_coherent"
DOMINANT_SHOT_NOISE = "shot_noise"


def select_strategy(
    dominant_error: str,
    target_accuracy: float,
    suggested_focus: Optional[list[str]] = None,
    base: Optional[Strategy] = None,
) -> Strategy:
    """Pick a minimal initial Strategy from the probe classification.

    Args:
        dominant_error: one of readout / gate_coherent / shot_noise.
        target_accuracy: desired absolute error (tighter targets justify more twirls).
        suggested_focus: optional VLM hint (techniques); used only to *add*, never
            to remove, and never to override the deterministic floor.
        base: optional Strategy to start from (defaults to a fresh one).

    Returns:
        A Strategy biased to the cheapest sufficient technique set.
    """
    strategy = base or Strategy()

    if dominant_error == DOMINANT_READOUT:
        techniques = [Technique.REM.value]                      # REM alone usually suffices
    elif dominant_error == DOMINANT_GATE_COHERENT:
        techniques = [Technique.REM.value, Technique.ZNE.value]  # readout + gate noise
    else:  # shot_noise (or unknown): mitigation won't help much; lean on shots
        techniques = [Technique.REM.value]

    # The VLM may *suggest* adding a technique, but cannot drop the REM floor.
    for t in suggested_focus or []:
        if t in (Technique.REM.value, Technique.PT.value, Technique.ZNE.value) and t not in techniques:
            techniques.append(t)

    strategy.techniques = techniques
    # Tighter targets get more twirls (lower variance), within reason.
    strategy.twirl_count = 16 if target_accuracy < 0.02 else 8
    return strategy


def decide(
    error_bar: float,
    error_estimate: Optional[float],
    target_accuracy: float,
    strategy: Strategy,
    vlm_verdict: Optional[dict[str, Any]] = None,
    confidence_threshold: float = 0.5,
) -> Decision:
    """Decide whether to STOP or how to retry, rules-first.

    Args:
        error_bar: the estimate's uncertainty (e.g. jackknife std).
        error_estimate: |estimate - reference| if a reference is available
            (e.g. during local benchmarking); otherwise None and the error_bar
            is used as the stopping proxy.
        target_accuracy: the stopping threshold.
        strategy: the current strategy (to compute escalation targets).
        vlm_verdict: optional structured VLM output (Phase L3). Recognized keys:
            extrapolation_monotone (bool), readout_anomaly (bool),
            improvement_meaningful (bool), confidence (float),
            recommended_action (str).
        confidence_threshold: below this VLM confidence, the verdict is ignored.

    Returns:
        A Decision whose ``action`` is STOP or a RETRY_* mode.
    """
    # The quantity we compare to target: a true error if known, else the bar.
    metric = error_estimate if error_estimate is not None else error_bar

    # 1. Target met -> stop early (no VLM needed). This is the efficiency win.
    if metric <= target_accuracy:
        return Decision(
            action=DecisionAction.STOP.value,
            reason=f"target met: {metric:.4f} <= {target_accuracy:.4f}",
            source="rules",
        )

    # 2. Consult the VLM verdict only if present and confident enough.
    use_vlm = bool(vlm_verdict) and vlm_verdict.get("confidence", 0.0) >= confidence_threshold
    if use_vlm:
        decision = _decide_from_vlm(vlm_verdict, strategy)
        if decision is not None:
            return decision

    # 3. Deterministic numeric fallback.
    return _decide_numeric(metric, target_accuracy, strategy)


def _decide_from_vlm(verdict: dict[str, Any], strategy: Strategy) -> Optional[Decision]:
    """Map a confident VLM verdict to a Decision; None if it has no opinion."""
    src = "vlm+rules"

    if verdict.get("readout_anomaly"):
        return Decision(
            action=DecisionAction.RETRY_CALIBRATION.value,
            reason="VLM: readout distribution anomalous -> recalibrate REM",
            source=src,
        )
    if verdict.get("extrapolation_monotone") is False and strategy.uses(Technique.ZNE.value):
        return Decision(
            action=DecisionAction.RETRY_STRATEGY.value,
            reason="VLM: ZNE extrapolation non-monotone -> change factory",
            invalidate=[],
            source=src,
        )
    if verdict.get("improvement_meaningful") is False:
        return Decision(
            action=DecisionAction.RETRY_STRATEGY.value,
            reason="VLM: improvement ~ shot noise -> reduce/switch technique set",
            source=src,
        )

    # An explicit recommendation, if the structured keys above didn't fire.
    rec = verdict.get("recommended_action")
    valid = {a.value for a in DecisionAction}
    if rec in valid:
        return Decision(action=rec, reason="VLM: explicit recommendation", source=src)
    return None


def _decide_numeric(metric: float, target: float, strategy: Strategy) -> Decision:
    """Pure numeric escalation when the target is unmet and no VLM opinion."""
    # Close to target -> just buy more shots (cheapest escalation).
    if metric <= _CLOSE_TO_TARGET_FACTOR * target:
        return Decision(
            action=DecisionAction.RETRY_SHOTS.value,
            reason=f"close to target ({metric:.4f} <= {_CLOSE_TO_TARGET_FACTOR}x{target:.4f}); more shots",
            source="rules",
        )
    # Far from target -> escalate the strategy (add a technique / change factory).
    return Decision(
        action=DecisionAction.RETRY_STRATEGY.value,
        reason=f"far from target ({metric:.4f} > {_CLOSE_TO_TARGET_FACTOR}x{target:.4f}); escalate strategy",
        source="rules",
    )


def escalate_strategy(strategy: Strategy) -> Strategy:
    """Produce the next, stronger strategy (used on RETRY_STRATEGY).

    Escalation order: add ZNE if absent -> add PT if absent -> advance the ZNE
    factory along Linear -> Exp -> Richardson.
    """
    techniques = list(strategy.techniques)
    if Technique.ZNE.value not in techniques:
        techniques.append(Technique.ZNE.value)
    elif Technique.PT.value not in techniques:
        techniques.append(Technique.PT.value)
    else:
        # All techniques present: advance the extrapolation factory.
        try:
            idx = FACTORY_ESCALATION.index(strategy.zne_factory)
            strategy.zne_factory = FACTORY_ESCALATION[min(idx + 1, len(FACTORY_ESCALATION) - 1)]
        except ValueError:
            strategy.zne_factory = FACTORY_ESCALATION[0]
    strategy.techniques = techniques
    return strategy
