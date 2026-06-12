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
    # SSE frames: progress/result/done/error. Queue-driven engine events (incl.
    # the experiment description) ride the "progress" frame with their real type
    # in data["event"] — the same shape the frontend consumes.
    events, result, experiment = [], None, None
    node_plots = {}  # node id -> list of streamed plot names
    with client.stream("POST", "/api/run", json=payload) as r:
        assert r.status_code == 200
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                data = line.split(":", 1)[1].strip()
                if event == "progress":
                    payload_data = json.loads(data)
                    inner = payload_data.get("event", "progress")
                    events.append(inner)
                    if inner == "experiment":
                        experiment = payload_data
                    elif inner == "node_done":
                        node_plots[payload_data["node"]] = [
                            p["name"] for p in payload_data.get("plots", [])
                        ]
                else:
                    events.append(event)
                    if event == "result":
                        result = json.loads(data)
    return events, result, experiment, node_plots


def test_run_stream_emits_progress_and_result():
    events, result, experiment, node_plots = _collect_events({
        "qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2",
        "use_vlm": False, "compare_baseline": True, "seed": 7,
    })
    counts = Counter(events)
    assert counts["node_done"] > 0     # live node progress streamed
    assert counts["decision"] > 0      # at least one validate decision streamed
    assert counts["experiment"] == 1   # experiment description streamed up front
    assert counts["result"] == 1
    assert counts["done"] == 1

    # The experiment-setup frame describes the actual problem (no shots needed).
    assert experiment is not None
    assert experiment["num_qubits"] == 2
    assert len(experiment["observable_terms"]) > 0
    assert "ideal" in experiment

    # The empirical_probe node streams the histograms the agent "sees".
    assert node_plots.get("empirical_probe") == ["readout_probe", "ghz_probe"]

    assert result is not None
    assert result["status"] in ("stopped", "exhausted")
    assert result["estimate"] is not None
    # The setup description is also embedded in the final payload.
    assert result["experiment"]["observable_terms"] == experiment["observable_terms"]
    # Figures the frontend renders are present.
    figs = result["figures"]
    assert "readout_probe" in figs and "ghz_probe" in figs
    assert "accuracy_vs_shots" in figs  # comparison requested
    # Audit trail surfaced.
    assert isinstance(result["audit"], list) and len(result["audit"]) > 0
