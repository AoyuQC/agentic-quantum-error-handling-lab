"""Structured-JSON VLM inspection tool — a Gateway-shaped seam.

Renders plot figures to base64 PNG, asks the VLM to return exactly one JSON
object matching a pydantic schema, and parses + validates it. On any failure
(no client, render error, non-JSON output, schema mismatch, low confidence) the
result is flagged ``degraded`` so the decision logic falls back to deterministic
numeric rules — the VLM never bypasses the rules floor or Policy.

Adapted from the NVIDIA blueprint's ``tools/vlm_tool.py`` sync/async bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from ..vlm.providers import VLMProvider
from ..vlm.renderer import render_plot_to_base64
from ..vlm.schemas import (
    PROBE_SCHEMA_HINT,
    VALIDATE_SCHEMA_HINT,
    ProbeClassification,
    ValidateDecision,
)

logger = logging.getLogger(__name__)


def _run_async(coro) -> Any:
    """Run an async coroutine from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop — run in a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _render_all(plots: list[dict[str, Any]]) -> list[str]:
    """Render plot records ({name, format, data}) to base64 PNG strings."""
    images: list[str] = []
    for plot in plots:
        fmt = plot.get("format")
        data = plot.get("data")
        if fmt == "plotly":
            images.append(render_plot_to_base64(data))
        elif fmt in ("png", "jpeg", "jpg") and isinstance(data, str):
            images.append(data)  # already base64
        else:
            logger.warning("skipping plot with unsupported format: %s", fmt)
    return images


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a possibly chatty VLM response."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def inspect_structured(
    vlm: Optional[VLMProvider],
    plots: list[dict[str, Any]],
    prompt: str,
    schema: Type[BaseModel],
    schema_hint: str,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Inspect plots with the VLM and return a validated dict (or a degraded flag).

    Returns:
        On success: ``{**model_fields, "degraded": False}``.
        On any failure: ``{"degraded": True, "reason": "..."}``.
    """
    if vlm is None:
        return {"degraded": True, "reason": "no VLM client configured"}

    try:
        images = _render_all(plots)
    except Exception as e:  # rendering requires kaleido
        logger.error("plot rendering failed: %s", e)
        return {"degraded": True, "reason": f"render failed: {e}"}
    if not images:
        return {"degraded": True, "reason": "no renderable plots"}

    full_prompt = (
        f"{prompt}\n\nRespond with EXACTLY ONE JSON object and nothing else, "
        f"matching this schema:\n{schema_hint}"
    )

    try:
        raw = _run_async(vlm.analyze_images(full_prompt, images))
    except Exception as e:
        logger.error("VLM call failed: %s", e)
        return {"degraded": True, "reason": f"vlm call failed: {e}"}

    parsed = _extract_json(raw if isinstance(raw, str) else str(raw))
    if parsed is None:
        return {"degraded": True, "reason": "response was not valid JSON"}

    try:
        model = schema.model_validate(parsed)
    except ValidationError as e:
        return {"degraded": True, "reason": f"schema validation failed: {e}"}

    result = model.model_dump()
    if result.get("confidence", 0.0) < confidence_threshold:
        result["degraded"] = True
        result["reason"] = "confidence below threshold"
    else:
        result["degraded"] = False
    return result


def classify_probe_with_vlm(
    vlm: Optional[VLMProvider],
    plots: list[dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """VLM classification of the probe histograms (ProbeClassification schema)."""
    prompt = (
        "You are inspecting measurement-outcome histograms from quantum error "
        "characterization probes. The first plot is a readout-calibration probe "
        "(state |0...0> was prepared; any mass off the all-zeros bar is readout "
        "error). The second is a GHZ probe (ideal mass sits only on |0...0> and "
        "|1...1>; extra mass elsewhere beyond readout error indicates gate / "
        "coherent error). Classify the dominant error source and suggest which "
        "mitigation techniques matter most."
    )
    return inspect_structured(
        vlm, plots, prompt, ProbeClassification, PROBE_SCHEMA_HINT, confidence_threshold
    )


def validate_with_vlm(
    vlm: Optional[VLMProvider],
    plots: list[dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """VLM judgment of the ZNE extrapolation plot (ValidateDecision schema)."""
    prompt = (
        "You are inspecting a zero-noise-extrapolation (ZNE) plot: measured "
        "expectation values vs noise scale factor, with the extrapolated "
        "zero-noise value marked. Judge whether the extrapolation is physically "
        "reasonable (monotone decay, no wild outliers), whether the readout "
        "distribution looks anomalous, and whether the improvement is meaningful "
        "versus shot noise. Recommend whether to stop or which retry mode to use."
    )
    return inspect_structured(
        vlm, plots, prompt, ValidateDecision, VALIDATE_SCHEMA_HINT, confidence_threshold
    )
