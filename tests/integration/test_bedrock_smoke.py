"""Bedrock smoke test — requires live AWS credentials + Bedrock model access.

Skipped by default (deselect with `-m "not bedrock"`); run explicitly with:
    pytest -m bedrock
It calls the real BedrockClaudeProvider on a probe-style figure and asserts the
response validates against the ProbeClassification schema.
"""

import os

import pytest

pytestmark = [pytest.mark.bedrock, pytest.mark.integration]


@pytest.mark.skipif(
    not os.environ.get("AQEM_RUN_BEDROCK"),
    reason="set AQEM_RUN_BEDROCK=1 (and AWS creds) to run the live Bedrock smoke test",
)
def test_bedrock_probe_classification_returns_valid_schema():
    from aqem.vlm import get_vlm_client
    from aqem.vlm.renderer import render_plot_to_base64  # noqa: F401  (ensures kaleido present)
    from aqem.tools.vlm_tool import classify_probe_with_vlm

    vlm = get_vlm_client({
        "provider": "bedrock",
        "model_id": os.environ.get("AQEM_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        "region": os.environ.get("AWS_REGION", "us-east-1"),
    })

    # A readout-error-looking histogram: mass leaking off |00>.
    fig = {
        "data": [{"type": "bar", "x": ["00", "01", "10", "11"], "y": [0.8, 0.08, 0.08, 0.04]}],
        "layout": {"title": {"text": "readout probe (prep |00>)"}},
    }
    result = classify_probe_with_vlm(
        vlm, [{"name": "readout", "format": "plotly", "data": fig}], confidence_threshold=0.0
    )
    assert not result.get("degraded"), result.get("reason")
    assert result["dominant_error"] in ("readout", "gate_coherent", "shot_noise")
