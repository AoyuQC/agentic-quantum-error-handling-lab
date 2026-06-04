"""Agentic Quantum Error Mitigation (AQEM) Lab.

A deterministic DAG-orchestrated agent that runs cheap probe circuits on Amazon
Braket, applies Mitiq error-mitigation techniques (REM / ZNE / PT), and uses a
VLM to inspect plots and drive PASS / adaptive-retry / early-stop decisions —
reaching a target accuracy with materially fewer shots than running the full
Mitiq stack blindly, under a deterministic Policy (controlled action set +
shot/cost budget).
"""

__version__ = "0.1.0"
