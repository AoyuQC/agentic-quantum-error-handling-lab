"""Amazon Bedrock AgentCore Runtime entrypoint.

Wraps the deterministic adaptive QEM loop as an AgentCore Runtime handler. The
loop, Policy, nodes, and Claude/Bedrock VLM run **in-process** (the tools in
``aqem.tools`` are the seams a Gateway MCP server would later route to). Large
artifacts (plots, audit, comparison) go to S3; the response carries the estimate,
shot ledger, and artifact URIs.

Local use (no AgentCore needed):
    from aqem.cloud.runtime import invoke
    invoke({"qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2"})

Served use:
    agentcore dev      # local hot-reload server
    agentcore deploy   # build + push + deploy to AgentCore Runtime
The module exposes ``app`` (a BedrockAgentCoreApp) when the SDK is installed.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..baseline.full_stack import BaselineConfig, run_full_stack_baseline
from ..loop import run_adaptive_loop
from ..models import Budget, Estimate
from ..problems import default_problem, ideal_expectation
from ..reporting.efficiency import compare
from ..reporting.plots import accuracy_vs_shots_figure
from .artifacts import make_artifact_store
from .guardrails import Guardrail


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _build_vlm(payload: dict[str, Any]):
    """Build the Bedrock Claude VLM client unless explicitly disabled."""
    if not payload.get("use_vlm", True):
        return None
    from ..vlm import get_vlm_client

    return get_vlm_client(
        {
            "provider": "bedrock",
            "model_id": payload.get("vlm_model_id")
            or _env("AQEM_VLM_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "region": payload.get("region") or _env("AWS_REGION", "us-east-1"),
            "temperature": 0,
            "max_tokens": 4096,
        }
    )


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one QEM session from an invocation payload and return a result dict.

    Payload keys (all optional):
        qubits (int)            : number of qubits (default 2)
        target_accuracy (float) : stopping threshold (default 0.06)
        device (str)            : noise-model device name (default qd_readout_2)
        budget_shots (int)      : shot budget (default 2_000_000)
        use_vlm (bool)          : use the Bedrock Claude VLM (default True)
        compare_baseline (bool) : also run the static baseline + comparison (default True)
        seed (int)              : RNG seed for reproducibility
        artifacts (str)         : artifact destination (s3://bucket/prefix or local path)
        guardrail_id (str)      : optional Bedrock guardrail id for text I/O
    """
    from ..config import resolve_device

    qubits = int(payload.get("qubits", 2))
    target = float(payload.get("target_accuracy", 0.06))
    device_name = payload.get("device") or _env("AQEM_DEVICE", "qd_readout_2")
    budget_shots = int(payload.get("budget_shots", 2_000_000))
    seed = payload.get("seed")
    run_id = str(payload.get("run_id") or _env("AQEM_RUN_ID", "run"))

    # Artifact store: S3 in the cloud (env or payload), local otherwise.
    artifacts_dest = payload.get("artifacts") or _env("AQEM_ARTIFACTS")
    region = payload.get("region") or _env("AWS_REGION", "us-east-1")
    store = make_artifact_store(artifacts_dest, run_id=run_id, region=region)

    # Optional managed Guardrails on the free-text task description (in).
    guardrail = Guardrail(
        guardrail_id=payload.get("guardrail_id") or _env("AQEM_GUARDRAIL_ID"),
        region=region,
    )
    task_text = payload.get("prompt", "")
    if task_text:
        gr = guardrail.check(task_text, source="INPUT")
        if not gr.allowed:
            return {"status": "blocked", "reason": gr.reason, "guardrail_action": gr.action}

    device = resolve_device(device_name)
    problem, circuit = default_problem(num_qubits=qubits, target_accuracy=target)
    ideal = ideal_expectation(circuit, problem.observable)
    vlm = _build_vlm(payload)

    run_cfg = {
        "probe_shots": int(payload.get("probe_shots", 2000)),
        "shot_per_base": int(payload.get("shot_per_base", 4000)),
        "overhead": int(payload.get("overhead", 3)),
        "rem_twirls": int(payload.get("rem_twirls", 20)),
        "use_ideal_for_validation": True,
        "vlm_confidence_threshold": float(payload.get("vlm_confidence_threshold", 0.5)),
    }

    record = run_adaptive_loop(
        problem, circuit, device, Budget(shots_total=budget_shots),
        config=run_cfg, vlm=vlm, seed=seed,
    )
    adaptive_est = record.final_outputs.get("estimate")

    response: dict[str, Any] = {
        "status": record.status,
        "iterations": record.iterations,
        "device": device_name,
        "ideal": ideal,
        "target_accuracy": target,
        "estimate": adaptive_est,
        "shots_used": record.final_outputs.get("shots_used"),
        "decision": record.final_outputs.get("decision"),
        "vlm_used": vlm is not None,
        "artifacts": {},
    }

    # Persist the audit trail.
    audit = record.final_outputs.get("audit", [])
    response["artifacts"]["audit"] = store.put_json("audit.json", audit)

    # Optional comparison vs the blind full-stack baseline.
    if payload.get("compare_baseline", True) and adaptive_est is not None:
        baseline_est = run_full_stack_baseline(problem, circuit, device, BaselineConfig())
        # Charge the adaptive side its FULL shot ledger (probe + calibration +
        # execution) so the comparison vs the (probe-free) baseline is honest.
        adaptive_total = Estimate.from_dict(adaptive_est)
        adaptive_total.shots_used = record.final_outputs.get("shots_used", adaptive_total.shots_used)
        cmp = compare(adaptive_total, baseline_est, ideal, target)
        response["comparison"] = cmp.to_dict()
        response["artifacts"]["comparison"] = store.put_json("comparison.json", cmp.to_dict())
        response["artifacts"]["accuracy_vs_shots"] = store.put_json(
            "accuracy_vs_shots.json", accuracy_vs_shots_figure(cmp)
        )

    # Guardrail the outgoing rationale text (out).
    decision_reason = (record.final_outputs.get("decision") or {}).get("reason", "")
    if decision_reason:
        gr = guardrail.check(decision_reason, source="OUTPUT")
        response["guardrail_output_action"] = gr.action

    return response


def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    """Plain function entry for local testing (no AgentCore SDK required)."""
    return handle(payload)


# --- AgentCore Runtime app (only if the SDK is installed) ------------------
try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def agent_entrypoint(payload: dict[str, Any]) -> dict[str, Any]:
        """AgentCore Runtime entrypoint — delegates to ``handle``."""
        return handle(payload or {})

except ImportError:  # SDK not installed (e.g. minimal local env)
    app = None


if __name__ == "__main__":
    if app is not None:
        app.run()
    else:  # pragma: no cover
        import json

        print(json.dumps(invoke({}), indent=2, default=str))
