"""Unit tests for the rules-first decision logic (no simulator)."""

from aqem.decision.rules import decide, escalate_strategy, select_strategy
from aqem.models import DecisionAction, Strategy, Technique


def test_select_strategy_readout_dominated_is_rem_only():
    s = select_strategy("readout", target_accuracy=0.05)
    assert s.techniques == [Technique.REM.value]


def test_select_strategy_gate_dominated_adds_zne():
    s = select_strategy("gate_coherent", target_accuracy=0.05)
    assert Technique.REM.value in s.techniques and Technique.ZNE.value in s.techniques


def test_select_strategy_tight_target_uses_more_twirls():
    loose = select_strategy("readout", target_accuracy=0.1)
    tight = select_strategy("readout", target_accuracy=0.01)
    assert tight.twirl_count > loose.twirl_count


def test_vlm_suggestion_can_add_but_not_drop_rem():
    s = select_strategy("readout", target_accuracy=0.05, suggested_focus=["ZNE"])
    assert Technique.REM.value in s.techniques  # floor preserved
    assert Technique.ZNE.value in s.techniques  # suggestion added


def test_decide_stops_when_target_met():
    d = decide(error_bar=0.01, error_estimate=0.01, target_accuracy=0.05, strategy=Strategy())
    assert d.action == DecisionAction.STOP.value


def test_decide_close_to_target_asks_for_shots():
    # error 0.07, target 0.05 -> within 2x -> retry_shots
    d = decide(error_bar=0.07, error_estimate=0.07, target_accuracy=0.05, strategy=Strategy())
    assert d.action == DecisionAction.RETRY_SHOTS.value


def test_decide_far_from_target_escalates_strategy():
    d = decide(error_bar=0.5, error_estimate=0.5, target_accuracy=0.05, strategy=Strategy())
    assert d.action == DecisionAction.RETRY_STRATEGY.value


def test_decide_uses_vlm_when_confident():
    verdict = {"readout_anomaly": True, "confidence": 0.9}
    d = decide(error_bar=0.5, error_estimate=0.5, target_accuracy=0.05,
               strategy=Strategy(), vlm_verdict=verdict)
    assert d.action == DecisionAction.RETRY_CALIBRATION.value
    assert d.source == "vlm+rules"


def test_decide_ignores_low_confidence_vlm():
    verdict = {"readout_anomaly": True, "confidence": 0.2}
    d = decide(error_bar=0.5, error_estimate=0.5, target_accuracy=0.05,
               strategy=Strategy(), vlm_verdict=verdict, confidence_threshold=0.5)
    # Falls back to numeric far-from-target rule.
    assert d.action == DecisionAction.RETRY_STRATEGY.value
    assert d.source == "rules"


def test_escalate_adds_zne_then_pt_then_advances_factory():
    s = Strategy(techniques=[Technique.REM.value], zne_factory="Linear")
    s = escalate_strategy(s)
    assert Technique.ZNE.value in s.techniques
    s = escalate_strategy(s)
    assert Technique.PT.value in s.techniques
    factory_before = s.zne_factory
    s = escalate_strategy(s)
    assert s.zne_factory != factory_before  # factory advanced
