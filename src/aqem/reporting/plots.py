"""Plotly figures for reports: ZNE extrapolation and accuracy-vs-shots.

Figures are plain dicts (Plotly JSON) — JSON-serializable for storage and
renderable to PNG via ``aqem.vlm.renderer.render_plot_to_base64`` for VLM
inspection.
"""

from __future__ import annotations

from typing import Any

from .efficiency import EfficiencyComparison


def zne_extrapolation_figure(
    zne_data: dict[str, float],
    extrapolated: float,
    ideal: float | None = None,
    title: str = "ZNE extrapolation",
) -> dict[str, Any]:
    """Scatter of measured expectation vs noise scale, plus the extrapolated
    zero-noise value at scale 0 (and the ideal reference if known).

    The VLM inspects this to judge whether the extrapolation is physically
    reasonable (monotone decay, no wild outliers).
    """
    scales = sorted(float(s) for s in zne_data)
    values = [zne_data[str(int(s)) if s.is_integer() else str(s)] for s in scales]

    data = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": scales,
            "y": values,
            "name": "measured",
            "marker": {"color": "#6366f1", "size": 9},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [0.0],
            "y": [extrapolated],
            "name": "extrapolated (zero-noise)",
            "marker": {"color": "#10b981", "size": 13, "symbol": "star"},
        },
    ]
    if ideal is not None:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0.0, max(scales) if scales else 1.0],
                "y": [ideal, ideal],
                "name": "ideal",
                "line": {"color": "#ef4444", "dash": "dash"},
            }
        )

    return {
        "data": data,
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "noise scale factor"}},
            "yaxis": {"title": {"text": "expectation value"}},
        },
    }


def accuracy_vs_shots_figure(
    comparison: EfficiencyComparison,
    title: str = "Accuracy vs shots — adaptive vs full-stack baseline",
) -> dict[str, Any]:
    """Overlay the adaptive trajectory (error vs cumulative shots) with the
    single baseline point and the target-accuracy threshold line."""
    traj = comparison.adaptive_trajectory or [comparison.adaptive]
    traj_sorted = sorted(traj, key=lambda p: p.shots)

    data = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "x": [p.shots for p in traj_sorted],
            "y": [p.error for p in traj_sorted],
            "name": "adaptive",
            "marker": {"color": "#10b981", "size": 9},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [comparison.baseline.shots],
            "y": [comparison.baseline.error],
            "name": "full-stack baseline",
            "marker": {"color": "#6366f1", "size": 13, "symbol": "square"},
        },
    ]

    max_shots = max(
        comparison.baseline.shots,
        max((p.shots for p in traj_sorted), default=comparison.adaptive.shots),
    )
    data.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": [0, max_shots],
            "y": [comparison.target_accuracy, comparison.target_accuracy],
            "name": "target accuracy",
            "line": {"color": "#ef4444", "dash": "dash"},
        }
    )

    return {
        "data": data,
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": "shots used"}},
            "yaxis": {"title": {"text": "absolute error |estimate - ideal|"}},
        },
    }
