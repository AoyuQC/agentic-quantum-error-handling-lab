"""Unit tests for the orchestration agent (the decider; rules are fallback).

Uses FakeVLM so the agent's Bedrock call is replaced by a canned JSON response —
no network/AWS. Covers: rules-fallback signalling (None), a reasoned STOP, a
retry that returns a sanitized next Strategy, the REM floor, and confidence
gating.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures.fake_vlm import FakeVLM  # noqa: E402

from aqem.decision.agent import propose_decision  # noqa: E402
from aqem.models import DecisionAction, Strategy, Technique  # noqa: E402


def _call(vlm, **overrides):
    kwargs = dict(
        target=0.01,
        current_error=0.08,
        current_error_bar=0.005,
        current_strategy=Strategy(),
        attempts=[],
        vlm_analysis=None,
        remaining_shots=1_000_000,
        iteration=1,
        max_iterations=8,
        confidence_threshold=0.5,
    )
    kwargs.update(overrides)
    return propose_decision(vlm, **kwargs)


def test_no_client_falls_back_to_rules():
    assert _call(None) is None


def test_non_json_falls_back_to_rules():
    assert _call(FakeVLM("I think you should keep going.")) is None


def test_low_confidence_falls_back_to_rules():
    vlm = FakeVLM({"action": "retry_strategy", "confidence": 0.2})
    assert _call(vlm) is None


def test_agent_stop_is_honored_with_reason():
    vlm = FakeVLM({
        "action": "stop",
        "reason": "plateaued at 0.08; target 0.01 infeasible on this noise model",
        "confidence": 0.9,
    })
    result = _call(vlm)
    assert result is not None
    decision, strategy = result
    assert decision.action == DecisionAction.STOP.value
    assert decision.source == "agent"
    assert "infeasible" in decision.reason
    assert strategy is None  # no next strategy on STOP


def test_agent_retry_returns_sanitized_strategy():
    vlm = FakeVLM({
        "action": "retry_strategy",
        "reason": "add ZNE for gate noise",
        "confidence": 0.8,
        "techniques": ["REM", "ZNE"],
        "zne_factory": "Exp",
        "zne_scale_factors": [1, 3, 5],
        "twirl_count": 16,
        "shot_per_base": 8000,
        "overhead": 4,
    })
    result = _call(vlm)
    assert result is not None
    decision, strategy = result
    assert decision.action == DecisionAction.RETRY_STRATEGY.value
    assert strategy is not None
    assert strategy.techniques == ["REM", "ZNE"]
    assert strategy.zne_factory == "Exp"
    assert strategy.shot_per_base == 8000
    assert strategy.twirl_count == 16


def test_agent_cannot_drop_rem_floor():
    vlm = FakeVLM({
        "action": "retry_strategy",
        "confidence": 0.9,
        "techniques": ["ZNE"],  # tries to drop REM
    })
    _, strategy = _call(vlm)
    assert Technique.REM.value in strategy.techniques


def test_richardson_factory_gets_enough_scale_points():
    vlm = FakeVLM({
        "action": "retry_strategy",
        "confidence": 0.9,
        "techniques": ["REM", "ZNE"],
        "zne_factory": "Richardson",
        "zne_scale_factors": [1, 3],  # too few for Richardson
    })
    _, strategy = _call(vlm)
    assert len(strategy.zne_scale_factors) >= 3


def test_scale_factors_always_start_at_one():
    vlm = FakeVLM({
        "action": "retry_strategy",
        "confidence": 0.9,
        "techniques": ["REM", "ZNE"],
        "zne_scale_factors": [3, 5],  # missing 1
    })
    _, strategy = _call(vlm)
    assert strategy.zne_scale_factors[0] == 1
