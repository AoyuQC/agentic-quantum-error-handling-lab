"""L3 acceptance: the adaptive loop runs end-to-end with a (fake) VLM, on a
local simulator, fully deterministic and offline. Verifies the VLM verdict is
consumed by the probe + validate nodes, and that a confident VLM steers the
decision while a degraded VLM falls back to the deterministic rules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.fake_vlm import FakeVLM  # noqa: E402

from aqem.loop import run_adaptive_loop  # noqa: E402
from aqem.models import Budget  # noqa: E402
from aqem.problems import default_problem  # noqa: E402

pytestmark = pytest.mark.integration


def _config():
    return {
        "probe_shots": 2000,
        "shot_per_base": 4000,
        "overhead": 3,
        "rem_twirls": 20,
        "use_ideal_for_validation": True,
        "vlm_confidence_threshold": 0.5,
    }


def test_loop_consumes_confident_vlm_probe_classification():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    # A confident readout classification — should drive REM-only and early-stop.
    vlm = FakeVLM({
        "dominant_error": "readout",
        "readout_asymmetry": True,
        "evidence": "off-state mass on readout probe",
        "suggested_focus": ["REM"],
        "confidence": 0.95,
    })

    record = run_adaptive_loop(
        problem, circuit, qd_readout_2, Budget(shots_total=2_000_000),
        config=_config(), vlm=vlm, seed=7,
    )

    probe = next(r for r in record.node_results if r.node_id == "empirical_probe")
    assert probe.outputs["classification"]["source"] == "vlm+rules"
    assert probe.outputs["vlm_classification"]["degraded"] is False
    assert record.status == "stopped"
    assert vlm.calls  # the VLM was actually invoked


def test_degraded_vlm_falls_back_to_rules():
    from aqem.braket_mitiq.noise_models import qd_readout_2

    problem, circuit = default_problem(num_qubits=2, target_accuracy=0.06)
    # Non-JSON -> degraded -> deterministic rules path must still succeed.
    vlm = FakeVLM("sorry, I can't analyze this")

    record = run_adaptive_loop(
        problem, circuit, qd_readout_2, Budget(shots_total=2_000_000),
        config=_config(), vlm=vlm, seed=7,
    )

    probe = next(r for r in record.node_results if r.node_id == "empirical_probe")
    # classification falls back to the rules source (not vlm+rules).
    assert probe.outputs["classification"].get("source", "rules") == "rules"
    assert record.status == "stopped"
