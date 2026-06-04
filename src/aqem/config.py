"""Configuration loading and device resolution.

Reads ``config/default.yaml`` (or a path) and resolves the named noise-model
device into a live Braket ``LocalSimulator``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root (…/agentic-quantum-error-handling-lab), three levels up from
# this file: src/aqem/config.py -> src/aqem -> src -> <root>.
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config; falls back to the packaged default."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_device(name: str):
    """Resolve a named noise-model device to a live LocalSimulator instance."""
    from . import braket_mitiq  # noqa: F401  (ensures subpackage import)
    from .braket_mitiq import noise_models

    device = getattr(noise_models, name, None)
    if device is None:
        available = [
            n for n in dir(noise_models)
            if not n.startswith("_") and n.startswith("qd_")
        ]
        raise ValueError(f"unknown device '{name}'. Available: {available}")
    return device


def run_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the loop's run-config knobs from the YAML structure."""
    probe = cfg.get("probe", {})
    baseline = cfg.get("baseline", {})
    vlm = cfg.get("vlm", {})
    return {
        "probe_shots": probe.get("shots", 2000),
        "shot_per_base": baseline.get("shot_per_base", 4000),
        "overhead": baseline.get("overhead", 3),
        "rem_twirls": probe.get("n_twirls", 20),
        "use_ideal_for_validation": True,
        "vlm_confidence_threshold": vlm.get("confidence_threshold", 0.5),
    }
