"""L5 acceptance: the FastAPI backend serves the API and streams a full run.

Offline (VLM disabled) and on the local simulator. Skipped if FastAPI isn't
installed (the `web` extra).
"""

import json
from collections import Counter

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aqem.web.server import app  # noqa: E402

pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health_and_devices():
    assert client.get("/api/health").json()["status"] == "ok"
    dev = client.get("/api/devices").json()
    assert "qd_readout_2" in dev["devices"]


def _collect_events(payload):
    events, result = [], None
    with client.stream("POST", "/api/run", json=payload) as r:
        assert r.status_code == 200
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                events.append(event)
                if event == "result":
                    result = json.loads(line.split(":", 1)[1].strip())
    return events, result


def test_run_stream_emits_progress_and_result():
    events, result = _collect_events({
        "qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2",
        "use_vlm": False, "compare_baseline": True, "seed": 7,
    })
    counts = Counter(events)
    assert counts["progress"] > 0      # live node/decision events streamed
    assert counts["result"] == 1
    assert counts["done"] == 1

    assert result is not None
    assert result["status"] in ("stopped", "exhausted")
    assert result["estimate"] is not None
    # Figures the frontend renders are present.
    figs = result["figures"]
    assert "readout_probe" in figs and "ghz_probe" in figs
    assert "accuracy_vs_shots" in figs  # comparison requested
    # Audit trail surfaced.
    assert isinstance(result["audit"], list) and len(result["audit"]) > 0
