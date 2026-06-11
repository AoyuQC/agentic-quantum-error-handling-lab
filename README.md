# Agentic Quantum Error Mitigation (AQEM) Lab

> VLM-guided, adaptive quantum error mitigation (Mitiq) on Amazon Braket, orchestrated as a deterministic DAG — reaching a target accuracy with **fewer shots** than running the full Mitiq stack blindly.

A deterministic DAG-orchestrated agent runs cheap **probe circuits** on a Braket
simulator, applies **Mitiq** techniques (REM / ZNE / PT), and uses a **VLM** to
inspect plots (probe histograms, ZNE extrapolation) to drive **PASS /
adaptive-retry / early-stop** decisions. Every side-effecting action passes a
deterministic **Policy** gate (controlled action set + shot/cost budget) and is
audited.

See [`docs/design-docs/design-doc.md`](docs/design-docs/design-doc.md) for the
full design, and the approved implementation plan for build phases.

## Status

Built **local-first**: the core runs entirely on a Braket `LocalSimulator` with
a noise model — no AWS infrastructure required. The cloud wrap (Bedrock
AgentCore Runtime + S3 artifacts + Guardrails) is implemented and runs the same
loop in-process; the VLM is managed Claude on Bedrock throughout.

| Phase | Scope | State |
|---|---|---|
| L0 | Scaffold + vendored Braket/Mitiq + VLM tools | ✅ done |
| L1 | Probe circuits + static baseline + efficiency harness | ✅ done |
| L2 | Deterministic DAG engine + Policy (rules-only) | ✅ done |
| L3 | Bedrock Claude VLM integration (structured JSON) | ✅ done |
| L4 | CLI + efficiency demo (adaptive vs baseline) | ✅ done |
| C5 | Cloud wrap: AgentCore Runtime + S3 + Guardrails | ✅ done |
| L5 | Web UI: FastAPI backend + React frontend | ✅ done |

### Efficiency demo

On a readout-dominated noise model, the adaptive loop reaches the target with
materially fewer shots than the blind full-stack baseline:

```
$ aqem report --device qd_readout_2 --target 0.06 --seed 7

=== Efficiency comparison: 2-qubit transverse-field Ising <H> estimate ===
  ideal reference        : 1.8041   (target accuracy 0.06)
                         shots     error            techniques
  baseline               84000    0.0870  ['REM', 'PT', 'ZNE']
  adaptive               36000    0.0246               ['REM']

  adaptive meets target  : True
  shots saved            : 48000
  shot ratio (base/adapt): 2.33x
  EFFICIENCY GAIN SHOWN   : True
```

The blind full stack wastes ZNE/PT shots (and is *less* accurate) where REM
alone suffices — exactly the inefficiency the adaptive loop avoids.

### Cloud (Bedrock AgentCore)

The same loop runs as an AgentCore **Runtime** entrypoint (`agent.py` →
`aqem.cloud.runtime`), with large artifacts in **S3** and optional Bedrock
**Guardrails** alongside the deterministic `policy/` layer. The VLM is managed
Claude on Bedrock. Run it locally without any AWS infra:

```python
from aqem.cloud.runtime import invoke
invoke({"qubits": 2, "target_accuracy": 0.06, "device": "qd_readout_2", "seed": 7})
```

Deploy to AWS (see [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md)):

```bash
pip install -e ".[cloud]"
export REGION=us-east-1
./deploy/deploy.sh provision     # S3 artifact bucket
./deploy/deploy.sh configure     # agentcore configure -e agent.py -n aqem
./deploy/deploy.sh deploy        # CodeBuild ARM64 image -> AgentCore Runtime
./deploy/deploy.sh invoke        # smoke test
```

### Web UI

A FastAPI + React console (in the spirit of the NVIDIA blueprint UI) drives a
run, streams **live per-node DAG progress**, and renders the probe histograms,
ZNE extrapolation, accuracy-vs-shots comparison, and Policy audit trail.

```bash
pip install -e ".[web]"
cd ui && npm install && npm run build && cd ..
aqem-web                      # http://localhost:8000 serves UI + API
# dev mode: `aqem-web` + `cd ui && npm run dev` (Vite on :3099)
```

See [`ui/README.md`](ui/README.md).

## Layout

```
src/aqem/
  braket_mitiq/   # vendored Braket+Mitiq primitives (Apache-2.0, see NOTICE)
  vlm/            # VLM provider abstraction (Bedrock-default) + plot rendering
  dag/            # deterministic DAG state machine + invalidation cascade
  nodes/          # the 8 QEM DAG stages (probe → … → validate → report)
  probes/         # probe circuits + histograms
  policy/         # action set + budget gate + audit log
  decision/       # rules-first + VLM decision logic
  baseline/       # static full-stack Mitiq baseline
  reporting/      # shots-vs-accuracy efficiency comparison
  tools/          # Gateway-shaped seams: braket_tool, vlm_tool
  cloud/          # C5: AgentCore Runtime entrypoint, S3 artifacts, Guardrails
  web/            # L5: FastAPI backend (SSE streaming) — `aqem-web`
ui/               # L5: React + Vite frontend (live DAG progress + charts)
config/           # default.yaml (VLM provider, noise model, budget, …)
deploy/           # Dockerfile target, deploy.sh, IAM policy, RUNBOOK.md
agent.py          # AgentCore Runtime entrypoint (BedrockAgentCoreApp)
tests/            # unit (offline) + integration (local simulator)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Offline unit tests + local-simulator integration tests (no AWS)
pytest tests/unit
pytest -m integration tests/integration
```

The default VLM provider is **managed Bedrock Claude** (`config/default.yaml`),
using the standard AWS credential chain. Swapping to the small NVIDIA
Ising-Calibration VLM on a SageMaker endpoint is a config-only change
(`vlm.provider: sagemaker`).

## Attribution

This project vendors Apache-2.0 code from
[amazon-braket-examples](https://github.com/amazon-braket/amazon-braket-examples)
and the [NVIDIA Quantum-Calibration-Agent-Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint).
See [`NOTICE`](NOTICE).
