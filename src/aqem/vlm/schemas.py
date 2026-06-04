"""Structured-output schemas for VLM responses.

The VLM is prompted to return exactly one JSON object matching one of these
pydantic models, so downstream handling is deterministic. On parse/validation
failure the caller flags the result ``degraded`` and the decision logic falls
back to deterministic numeric rules (the VLM never bypasses Policy or the rules
floor).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProbeClassification(BaseModel):
    """VLM classification of the empirical-probe histograms."""

    dominant_error: Literal["readout", "gate_coherent", "shot_noise"] = Field(
        description="Which error source dominates the probe histograms."
    )
    readout_asymmetry: bool = Field(
        default=False,
        description="Whether readout error looks asymmetric across qubits / 0<->1.",
    )
    evidence: str = Field(
        default="",
        description="What in the histograms supports this classification.",
    )
    suggested_focus: list[Literal["REM", "PT", "ZNE"]] = Field(
        default_factory=list,
        description="Mitigation techniques the probe suggests prioritizing.",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this classification."
    )


class ValidateDecision(BaseModel):
    """VLM judgment of the post-mitigation result / ZNE extrapolation plot."""

    extrapolation_monotone: bool = Field(
        default=True,
        description="Whether the ZNE points decay monotonically with noise scale.",
    )
    has_outliers: bool = Field(
        default=False, description="Whether any ZNE point is a clear outlier."
    )
    readout_anomaly: bool = Field(
        default=False,
        description="Whether the readout distribution looks anomalous / drifted.",
    )
    improvement_meaningful: bool = Field(
        default=True,
        description="Whether the mitigation improvement exceeds shot noise.",
    )
    recommended_action: Literal[
        "stop", "retry_shots", "retry_calibration", "retry_strategy"
    ] = Field(default="stop", description="Recommended next action.")
    rationale: str = Field(default="", description="Short justification.")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this judgment."
    )


# JSON-schema strings embedded into prompts so the model knows the exact shape.
PROBE_SCHEMA_HINT = (
    '{"dominant_error": "readout|gate_coherent|shot_noise", '
    '"readout_asymmetry": true|false, "evidence": "string", '
    '"suggested_focus": ["REM"|"PT"|"ZNE", ...], "confidence": 0.0-1.0}'
)

VALIDATE_SCHEMA_HINT = (
    '{"extrapolation_monotone": true|false, "has_outliers": true|false, '
    '"readout_anomaly": true|false, "improvement_meaningful": true|false, '
    '"recommended_action": "stop|retry_shots|retry_calibration|retry_strategy", '
    '"rationale": "string", "confidence": 0.0-1.0}'
)
