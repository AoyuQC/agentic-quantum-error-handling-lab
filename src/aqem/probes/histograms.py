"""Plotly histograms of probe-circuit measurement outcomes.

Figures are emitted as Plotly JSON (dicts) so they are JSON-serializable for
audit/report storage and can be rendered to base64 PNG by
``aqem.vlm.renderer.render_plot_to_base64`` before being sent to the VLM.
"""

from __future__ import annotations

from typing import Any


def _normalize(counts: dict[str, int | float]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def histogram_figure(
    counts: dict[str, int | float],
    title: str = "probe",
    highlight: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Plotly bar-chart figure (as a dict) of an outcome distribution.

    Args:
        counts: bitstring -> count (or probability) mapping.
        title: figure title.
        highlight: bitstrings to color distinctly (e.g. the ideal outcomes);
            useful for the VLM to see where mass *should* concentrate.

    Returns:
        Plotly figure as a plain dict (JSON-serializable).
    """
    probs = _normalize(counts)
    keys = sorted(probs.keys())
    highlight = set(highlight or [])
    colors = ["#10b981" if k in highlight else "#6366f1" for k in keys]

    return {
        "data": [
            {
                "type": "bar",
                "x": keys,
                "y": [probs[k] for k in keys],
                "marker": {"color": colors},
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "bitstring"}, "type": "category"},
            "yaxis": {"title": {"text": "probability"}, "range": [0, 1]},
            "bargap": 0.05,
        },
    }
