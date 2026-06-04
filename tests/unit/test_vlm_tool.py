"""Unit tests for VLM schemas and the structured-JSON vlm_tool (offline).

Uses FakeVLM so no network / AWS is involved; exercises JSON extraction,
schema validation, and the graceful degradation paths.
"""

import sys
from pathlib import Path

import pytest

# Make the FakeVLM fixture importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.fake_vlm import FakeVLM  # noqa: E402

from aqem.vlm.schemas import ProbeClassification, ValidateDecision  # noqa: E402
from aqem.tools.vlm_tool import (  # noqa: E402
    _extract_json,
    classify_probe_with_vlm,
    validate_with_vlm,
)

# A trivial renderable plotly figure.
_FIG = {"data": [{"type": "bar", "x": ["00", "11"], "y": [0.5, 0.5]}], "layout": {}}
_PLOTS = [{"name": "p", "format": "plotly", "data": _FIG}]


def test_schema_rejects_bad_enum():
    with pytest.raises(Exception):
        ProbeClassification(dominant_error="cosmic_rays", confidence=0.9)


def test_validate_schema_defaults():
    v = ValidateDecision(recommended_action="stop", confidence=0.8)
    assert v.extrapolation_monotone is True
    assert v.has_outliers is False


@pytest.mark.parametrize(
    "text",
    [
        '{"dominant_error": "readout", "confidence": 0.9}',
        '```json\n{"dominant_error": "readout", "confidence": 0.9}\n```',
        'Here is my analysis: {"dominant_error": "readout", "confidence": 0.9}. Done.',
    ],
)
def test_extract_json_handles_chatty_and_fenced(text):
    parsed = _extract_json(text)
    assert parsed["dominant_error"] == "readout"


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("no json here at all") is None


def test_classify_probe_with_confident_vlm():
    vlm = FakeVLM({
        "dominant_error": "readout",
        "readout_asymmetry": True,
        "evidence": "mass off |00>",
        "suggested_focus": ["REM"],
        "confidence": 0.92,
    })
    result = classify_probe_with_vlm(vlm, _PLOTS)
    assert result["degraded"] is False
    assert result["dominant_error"] == "readout"
    assert vlm.calls and vlm.calls[0]["n_images"] == 1


def test_low_confidence_is_degraded():
    vlm = FakeVLM({"dominant_error": "readout", "confidence": 0.1})
    result = classify_probe_with_vlm(vlm, _PLOTS, confidence_threshold=0.5)
    assert result["degraded"] is True
    assert "confidence" in result["reason"]


def test_non_json_response_is_degraded():
    vlm = FakeVLM("I cannot help with that.")
    result = validate_with_vlm(vlm, _PLOTS)
    assert result["degraded"] is True


def test_no_client_is_degraded():
    result = classify_probe_with_vlm(None, _PLOTS)
    assert result["degraded"] is True
    assert "no VLM client" in result["reason"]


def test_schema_mismatch_is_degraded():
    # valid JSON but wrong field type -> validation fails -> degraded
    vlm = FakeVLM('{"dominant_error": "readout", "confidence": "high"}')
    result = classify_probe_with_vlm(vlm, _PLOTS)
    assert result["degraded"] is True
