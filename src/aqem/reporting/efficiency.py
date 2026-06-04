"""Efficiency comparison: accuracy vs shots, adaptive loop vs static baseline.

The headline metric is the **shots-to-target ratio** — how many shots each
approach spent to reach (equal or better) accuracy against the exact reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..models import Estimate


@dataclass
class AccuracyPoint:
    """One (shots, accuracy) sample produced by an approach.

    ``error`` is the absolute deviation from the exact reference, |value - ideal|.
    """

    label: str
    shots: int
    value: float
    error: float
    error_bar: float = 0.0
    techniques: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def accuracy_point(label: str, estimate: Estimate, ideal: float) -> AccuracyPoint:
    """Score an Estimate against the exact reference value."""
    return AccuracyPoint(
        label=label,
        shots=estimate.shots_used,
        value=estimate.value,
        error=abs(estimate.value - ideal),
        error_bar=estimate.error_bar,
        techniques=list(estimate.techniques),
    )


@dataclass
class EfficiencyComparison:
    """The adaptive-vs-baseline efficiency result."""

    ideal: float
    target_accuracy: float
    adaptive: AccuracyPoint
    baseline: AccuracyPoint
    adaptive_trajectory: list[AccuracyPoint] = field(default_factory=list)

    @property
    def adaptive_meets_target(self) -> bool:
        return self.adaptive.error <= self.target_accuracy

    @property
    def baseline_meets_target(self) -> bool:
        return self.baseline.error <= self.target_accuracy

    @property
    def shot_ratio(self) -> Optional[float]:
        """baseline_shots / adaptive_shots (>1 means adaptive is cheaper)."""
        if self.adaptive.shots <= 0:
            return None
        return self.baseline.shots / self.adaptive.shots

    @property
    def shots_saved(self) -> int:
        return self.baseline.shots - self.adaptive.shots

    @property
    def efficiency_gain_demonstrated(self) -> bool:
        """Adaptive hit the target using strictly fewer shots, at accuracy no
        worse than the baseline (within one combined error bar of slack)."""
        slack = self.adaptive.error_bar + self.baseline.error_bar
        return (
            self.adaptive_meets_target
            and self.adaptive.shots < self.baseline.shots
            and self.adaptive.error <= self.baseline.error + slack
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ideal": self.ideal,
            "target_accuracy": self.target_accuracy,
            "adaptive": self.adaptive.to_dict(),
            "baseline": self.baseline.to_dict(),
            "adaptive_trajectory": [p.to_dict() for p in self.adaptive_trajectory],
            "shot_ratio": self.shot_ratio,
            "shots_saved": self.shots_saved,
            "adaptive_meets_target": self.adaptive_meets_target,
            "baseline_meets_target": self.baseline_meets_target,
            "efficiency_gain_demonstrated": self.efficiency_gain_demonstrated,
        }


def compare(
    adaptive: Estimate,
    baseline: Estimate,
    ideal: float,
    target_accuracy: float,
    adaptive_trajectory: list[AccuracyPoint] | None = None,
) -> EfficiencyComparison:
    """Build the efficiency comparison from two estimates and the reference."""
    return EfficiencyComparison(
        ideal=ideal,
        target_accuracy=target_accuracy,
        adaptive=accuracy_point("adaptive", adaptive, ideal),
        baseline=accuracy_point("baseline", baseline, ideal),
        adaptive_trajectory=adaptive_trajectory or [],
    )
