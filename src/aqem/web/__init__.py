"""Web UI backend (FastAPI) for the AQEM lab.

Exposes the adaptive QEM loop over HTTP with Server-Sent-Events streaming of
live per-node progress, plus the Plotly figures and audit trail consumed by the
React frontend in ``ui/``.
"""

from .server import app, main

__all__ = ["app", "main"]
