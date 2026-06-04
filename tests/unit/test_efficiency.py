"""Unit tests for the efficiency comparison logic (no simulator)."""

from aqem.models import Estimate
from aqem.reporting.efficiency import accuracy_point, compare
from aqem.reporting.plots import accuracy_vs_shots_figure, zne_extrapolation_figure


def _est(value, shots, error_bar=0.0, techniques=None):
    return Estimate(value=value, error_bar=error_bar, shots_used=shots,
                    techniques=techniques or [])


def test_accuracy_point_scores_against_ideal():
    p = accuracy_point("x", _est(1.05, 1000), ideal=1.0)
    assert abs(p.error - 0.05) < 1e-9
    assert p.shots == 1000


def test_efficiency_gain_demonstrated_when_adaptive_cheaper_and_accurate():
    # adaptive: hits target (err 0.02) with 10k shots; baseline: err 0.03 with 100k.
    adaptive = _est(1.02, 10_000, techniques=["REM"])
    baseline = _est(1.03, 100_000, techniques=["REM", "PT", "ZNE"])
    cmp = compare(adaptive, baseline, ideal=1.0, target_accuracy=0.05)

    assert cmp.adaptive_meets_target
    assert cmp.efficiency_gain_demonstrated
    assert cmp.shots_saved == 90_000
    assert cmp.shot_ratio == 10.0


def test_no_gain_when_adaptive_misses_target():
    adaptive = _est(1.30, 10_000)       # error 0.30 > target
    baseline = _est(1.01, 100_000)
    cmp = compare(adaptive, baseline, ideal=1.0, target_accuracy=0.05)
    assert not cmp.adaptive_meets_target
    assert not cmp.efficiency_gain_demonstrated


def test_no_gain_when_adaptive_not_cheaper():
    adaptive = _est(1.01, 100_000)
    baseline = _est(1.01, 100_000)
    cmp = compare(adaptive, baseline, ideal=1.0, target_accuracy=0.05)
    assert not cmp.efficiency_gain_demonstrated  # not strictly fewer shots


def test_plot_figures_are_serializable_dicts():
    adaptive = _est(1.02, 10_000)
    baseline = _est(1.03, 100_000)
    cmp = compare(adaptive, baseline, ideal=1.0, target_accuracy=0.05)

    fig1 = accuracy_vs_shots_figure(cmp)
    fig2 = zne_extrapolation_figure({"1": 1.7, "3": 1.5, "7": 1.2}, extrapolated=1.9, ideal=1.8)
    assert "data" in fig1 and "layout" in fig1
    assert "data" in fig2 and fig2["data"][1]["x"] == [0.0]
