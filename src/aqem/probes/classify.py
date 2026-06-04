"""Deterministic probe-histogram classifier (rules fallback for empirical_probe).

This is the numeric baseline the VLM augments in Phase L3. From the readout and
GHZ probe outcome distributions it estimates which error source dominates:

  - readout       : the all-zeros readout probe has substantial mass off |0...0>,
                    indicating measurement bit-flips.
  - gate_coherent : the GHZ probe leaks mass onto bitstrings other than the two
                    ideal peaks (|0...0>, |1...1>) beyond what readout alone explains.
  - shot_noise    : neither readout nor gate error is appreciable; remaining
                    deviation is sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbeClassification:
    """Result of classifying the probe histograms."""

    dominant_error: str            # "readout" | "gate_coherent" | "shot_noise"
    readout_error: float           # mass off the prepared |0...0> state
    ghz_leakage: float             # GHZ mass off the two ideal peaks
    suggested_focus: list[str]
    confidence: float
    source: str = "rules"

    def to_dict(self) -> dict:
        return {
            "dominant_error": self.dominant_error,
            "readout_error": self.readout_error,
            "ghz_leakage": self.ghz_leakage,
            "suggested_focus": self.suggested_focus,
            "confidence": self.confidence,
            "source": self.source,
        }


def _normalize(counts: dict[str, int | float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def classify_probes(
    readout_counts: dict[str, int | float],
    ghz_counts: dict[str, int | float],
    num_qubits: int,
    readout_threshold: float = 0.02,
    gate_threshold: float = 0.05,
) -> ProbeClassification:
    """Classify the dominant error source from the two probe histograms."""
    ro = _normalize(readout_counts)
    ghz = _normalize(ghz_counts)

    zeros = "0" * num_qubits
    ones = "1" * num_qubits

    # Readout error: probability mass NOT on the prepared all-zeros state.
    readout_error = 1.0 - ro.get(zeros, 0.0)

    # GHZ leakage: mass off the two ideal peaks. Some of this is itself readout
    # error; subtract a rough readout allowance so the remainder flags gate error.
    ideal_mass = ghz.get(zeros, 0.0) + ghz.get(ones, 0.0)
    ghz_leakage = max(0.0, 1.0 - ideal_mass)
    gate_excess = max(0.0, ghz_leakage - readout_error)

    if gate_excess >= gate_threshold:
        dominant = "gate_coherent"
        focus = ["REM", "ZNE"]
        confidence = min(1.0, 0.5 + gate_excess)
    elif readout_error >= readout_threshold:
        dominant = "readout"
        focus = ["REM"]
        confidence = min(1.0, 0.5 + readout_error)
    else:
        dominant = "shot_noise"
        focus = ["REM"]
        confidence = 0.6

    return ProbeClassification(
        dominant_error=dominant,
        readout_error=round(readout_error, 4),
        ghz_leakage=round(ghz_leakage, 4),
        suggested_focus=focus,
        confidence=round(confidence, 3),
    )
