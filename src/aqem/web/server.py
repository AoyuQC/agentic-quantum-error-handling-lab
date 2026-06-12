"""FastAPI backend for the AQEM web UI.

Mirrors the NVIDIA blueprint's server pattern (FastAPI + CORS + streaming) but
right-sized for this project. Endpoints:

    GET  /api/health                  liveness
    GET  /api/devices                 available noise-model devices
    POST /api/run        (SSE)        run the adaptive loop, streaming progress
                                       events and a final result payload

The run executes in a worker thread; the engine ``observer`` pushes small JSON
events onto a queue that the SSE generator drains, so the browser sees live
per-node progress. The final event carries the estimate, the Plotly figures
(probe histograms, ZNE extrapolation, accuracy-vs-shots), the efficiency
comparison, and the Policy audit trail.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..baseline.full_stack import BaselineConfig, run_full_stack_baseline
from ..loop import run_adaptive_loop
from ..models import Budget, Estimate
from ..problems import default_problem, ideal_expectation
from ..reporting.efficiency import compare
from ..reporting.plots import accuracy_vs_shots_figure, zne_extrapolation_figure

app = FastAPI(title="AQEM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Named local noise-model devices (no AWS needed).
DEVICES = ["qd_readout", "qd_readout_2", "qd_depol", "qd_total", "qd_amp"]

_SENTINEL = object()


class RunRequest(BaseModel):
    qubits: int = 2
    target_accuracy: float = 0.06
    device: str = "qd_readout_2"
    budget_shots: int = 2_000_000
    use_vlm: bool = False
    compare_baseline: bool = True
    seed: int | None = 7


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/devices")
def devices() -> dict[str, Any]:
    return {"devices": DEVICES, "default": "qd_readout_2"}


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _experiment(problem, circuit, ideal: float, req: RunRequest) -> dict[str, Any]:
    """Describe the experiment the agent is solving, for the UI setup panel.

    All values are read off the already-constructed Problem/Circuit — no shots
    are consumed (``ideal`` is the exact noiseless reference).
    """
    return {
        "num_qubits": problem.num_qubits,
        "description": problem.description,
        "observable_terms": [[float(c), p] for c, p in problem.observable],
        "ansatz": f"depth {circuit.depth}, {len(circuit.instructions)} gates",
        "ideal": ideal,
        "target_accuracy": req.target_accuracy,
        "device": req.device,
        "budget_shots": req.budget_shots,
        "seed": req.seed,
    }


def _run_stream(req: RunRequest):
    """Generator yielding SSE frames for one adaptive run."""
    from ..config import resolve_device

    events: "queue.Queue[Any]" = queue.Queue()

    def observer(ev: dict) -> None:
        events.put(ev)

    result_box: dict[str, Any] = {}

    def worker() -> None:
        try:
            device = resolve_device(req.device)
            problem, circuit = default_problem(req.qubits, target_accuracy=req.target_accuracy)
            ideal = ideal_expectation(circuit, problem.observable)

            # Describe the experiment up front so the UI setup panel populates
            # immediately (the same observer the engine uses).
            observer({"event": "experiment", **_experiment(problem, circuit, ideal, req)})

            vlm = None
            if req.use_vlm:
                from ..vlm import get_vlm_client

                vlm = get_vlm_client({"provider": "bedrock"})

            record = run_adaptive_loop(
                problem, circuit, device, Budget(shots_total=req.budget_shots),
                config={
                    "probe_shots": 2000, "shot_per_base": 4000, "overhead": 3,
                    "rem_twirls": 20, "use_ideal_for_validation": True,
                },
                vlm=vlm, seed=req.seed, observer=observer,
            )

            payload = _build_result(record, problem, circuit, device, ideal, req)
            result_box["result"] = payload
        except Exception as e:  # surface errors to the client
            result_box["error"] = str(e)
        finally:
            events.put(_SENTINEL)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        item = events.get()
        if item is _SENTINEL:
            break
        yield _sse("progress", item)

    if "error" in result_box:
        yield _sse("error", {"message": result_box["error"]})
    else:
        yield _sse("result", result_box["result"])
    yield _sse("done", {})


def _build_result(record, problem, circuit, device, ideal, req: RunRequest) -> dict[str, Any]:
    """Assemble the final result payload, including Plotly figures."""
    from ..probes.histograms import histogram_figure

    est_dict = record.final_outputs.get("estimate")
    figures: dict[str, Any] = {}

    # Probe histograms (from the empirical_probe node result).
    probe = next((r for r in record.node_results if r.node_id == "empirical_probe"), None)
    if probe is not None:
        counts = probe.outputs.get("counts", {})
        nq = problem.num_qubits
        zeros, ones = "0" * nq, "1" * nq
        if "readout" in counts:
            figures["readout_probe"] = histogram_figure(
                counts["readout"], "Readout probe (prep |0…0>)", [zeros]
            )
        if "ghz" in counts:
            figures["ghz_probe"] = histogram_figure(counts["ghz"], "GHZ probe", [zeros, ones])
        figures["classification"] = probe.outputs.get("classification")

    # ZNE extrapolation figure (when ZNE ran).
    if est_dict and len(est_dict.get("zne_data", {})) >= 2:
        figures["zne"] = zne_extrapolation_figure(
            est_dict["zne_data"], est_dict["value"], ideal=ideal
        )

    result: dict[str, Any] = {
        "status": record.status,
        "iterations": record.iterations,
        "experiment": _experiment(problem, circuit, ideal, req),
        "device": req.device,
        "ideal": ideal,
        "target_accuracy": req.target_accuracy,
        "estimate": est_dict,
        "shots_used": record.final_outputs.get("shots_used"),
        "decision": record.final_outputs.get("decision"),
        "audit": record.final_outputs.get("audit", []),
        "figures": figures,
        "vlm_used": req.use_vlm,
    }

    if req.compare_baseline and est_dict is not None:
        baseline_est = run_full_stack_baseline(problem, circuit, device, BaselineConfig())
        adaptive_total = Estimate.from_dict(est_dict)
        adaptive_total.shots_used = record.final_outputs.get("shots_used", adaptive_total.shots_used)
        cmp = compare(adaptive_total, baseline_est, ideal, req.target_accuracy)
        result["comparison"] = cmp.to_dict()
        result["figures"]["accuracy_vs_shots"] = accuracy_vs_shots_figure(cmp)

    return result


@app.post("/api/run")
def run(req: RunRequest) -> StreamingResponse:
    return StreamingResponse(_run_stream(req), media_type="text/event-stream")


# Serve the built frontend (if present) so a single process serves UI + API.
_UI_DIST = Path(__file__).resolve().parents[3] / "ui" / "dist"
if _UI_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
