# Implementation To-Do — Agentic QEM Lab

Tracks the build of the [design doc](design-doc.md). Build order is **local-first
/ Claude-first**: phases L0–L4 run entirely on a Braket `LocalSimulator` (no AWS),
with clean seams so the cloud wrap (C5) adds on without a rewrite.

**Key decisions:** deterministic DAG state machine (not LLM-driven); VLM is
provider-pluggable, default = managed Bedrock Claude, with the small NVIDIA
Ising VLM on SageMaker as a later config-only swap.

Legend: `[x]` done · `[ ]` pending · `[~]` deferred

---

## L0 — Scaffold + reuse import  `[x]`
- [x] `src/aqem` package skeleton; `pyproject.toml` + `requirements.txt` (pinned); fresh venv
- [x] Vendor Braket+Mitiq layer into `braket_mitiq/` (Apache-2.0 headers + `NOTICE`):
      `mitiq_braket_tools`, `mitigation_tools`, `program_set_tools`, `observable_tools`,
      `circuit_tools` (metadata-free helpers only), `noise_models`
- [x] Vendor VLM `renderer.py` + `providers.py` (default flipped to `bedrock`)
- [x] `config/default.yaml`, README, `.gitignore`
- **Accept:** ProgramSet runs on `qd_total` → counts; kaleido renders plotly → base64 PNG ✅
- Notes: pinned `langchain-core>=1.4` (langchain-aws 1.5 needs the 1.x line); `kaleido==0.2.1` (bundled chromium, headless)

## L1 — Probe + static baseline + harness  `[x]`
- [x] `models.py`: `Problem`, `Budget`, `Estimate`
- [x] `problems.py`: Ising Hamiltonian, ansatz, exact noiseless reference (`ideal_expectation`)
- [x] `probes/circuits.py` (readout + GHZ/Bell), `probes/histograms.py` (plotly)
- [x] `baseline/full_stack.py`: full REM+PT+ZNE stack, exact shot accounting, jackknife error bar
- [x] `reporting/efficiency.py` + `reporting/plots.py`
- **Accept:** baseline produces estimate + shot count; harness scores vs noiseless reference ✅
- Notes: added `cirq-ionq==1.6.1` (mitiq Braket↔cirq conversion dep)

## L2 — Deterministic DAG engine + Policy (rules-only)  `[x]`
- [x] `models.py` extensions: `Strategy`, `Calibration`, `Decision`, `NodeResult`, `Technique`
- [x] `policy/`: `Action` enum (controlled set), budget hard gate, no-recalibration guard,
      retry cap, append-only JSONL audit
- [x] `decision/rules.py`: rules-first `select_strategy` / `decide` / `escalate_strategy`
- [x] `dag/`: `Node` ABC, `RunContext`, `DAGEngine` (topo order, cycle check, cascading invalidation)
- [x] 8 nodes: `empirical_probe → strategy_select → readout_calibrate → circuit_generate →
      execute → post_process → validate → report`; `tools/braket_tool.py`; `probes/classify.py`; `loop.py`
- **Accept:** invalidation cascade proven; budget + action-set gates proven; loop early-stops on local sim ✅

## L3 — Bedrock Claude VLM integration (structured JSON)  `[x]`
- [x] `vlm/providers.py`: `BedrockClaudeProvider` (default, ChatBedrockConverse) + `SageMakerProvider`
- [x] `vlm/schemas.py`: pydantic `ProbeClassification`, `ValidateDecision`
- [x] `tools/vlm_tool.py`: render → prompt-for-JSON → validate → graceful degrade to rules
- [x] Wire VLM verdict into `empirical_probe` (classification) and `validate` (decision)
- **Accept:** confident VLM steers; degraded VLM falls back to rules; loop deterministic offline via `FakeVLM` ✅
- [x] `@pytest.mark.bedrock` live smoke test (skipped without `AQEM_RUN_BEDROCK` + creds)

## L4 — CLI + efficiency demo  `[x]`
- [x] `config.py` (YAML load + device resolution); `cli.py`: `aqem baseline | run | report`
- [x] `report` runs baseline + adaptive, prints shots-vs-accuracy comparison, writes figures/JSON
- [x] Bug fix: `Policy(audit or AuditLog())` discarded empty passed logs (`__len__` falsiness) → `is not None`
- **Accept:** on `qd_readout_2`, adaptive hits target with fewer shots than full-stack baseline ✅
- **Demo:** baseline 84k shots / err 0.087 (REM+PT+ZNE) vs adaptive 36k / err 0.025 (REM) → **2.33× fewer shots, better accuracy**

---

## C5 — Cloud wrap  `[x]`
Wrap the local-first loop for Amazon Bedrock AgentCore. **VLM stays managed
Claude on Bedrock** (the SageMaker/Ising-VLM swap is dropped from scope). Code
has clean seams (in-process `tools/`, JSON-serializable models).
- [x] Wrap the loop as a Bedrock AgentCore **Runtime** entrypoint (`cloud/runtime.py`, `agent.py`)
- [x] Move large artifacts (arrays, plots) to **S3** (`cloud/artifacts.py`, with local fallback)
- [x] Add **Bedrock Guardrails** alongside the existing `policy/` layer (`cloud/guardrails.py`)
- [x] AgentCore Observability: structured run summary + full audit returned in the response and persisted
- [x] Deployment assets: `Dockerfile`, `deploy/deploy.sh`, `deploy/execution-role-policy.json`, `deploy/RUNBOOK.md`
- [x] Live Bedrock VLM verified end-to-end (fixed langchain-aws image-block format → `source_type/data/mime_type`)
- [x] Live S3 artifact path verified (bucket `aqem-artifacts-<acct>-us-east-1` provisioned; write/list confirmed)
- **Accept:** ✅ Runtime handler runs locally (`invoke(...)` / `agentcore dev`); live Bedrock VLM + S3 verified;
  efficiency gain reproduced through the cloud handler.
- **Operator-run (needs an interactive session / IAM):** `agentcore configure` → `deploy` (creates the
  execution role + ECR + Runtime). Fully scripted in `deploy/deploy.sh` + `deploy/RUNBOOK.md`.
- Tools exposed in-process for now; a Gateway MCP server is a thin future add (seams already in `tools/`).

---

## Future work (design-doc §9)
- [ ] Real-device execution: Braket local device emulator + QPUs (empirical-probe design carries over)
- [ ] Neural QEC decoding (NVIDIA Ising-Decoding) on syndrome data
- [ ] Richer characterization: optionally fold in device calibration metadata when accessible

---

## Verification
- Unit (offline, no AWS): `pytest tests/unit` — Policy gates, DAG invalidation, decision rules,
  VLM schemas/degradation (deterministic via `tests/fixtures/fake_vlm.py`)
- Integration (local sim, no AWS): `pytest -m integration` — ProgramSet execution, baseline,
  adaptive loop, baseline-vs-adaptive efficiency, CLI commands
- Live Bedrock smoke: `AQEM_RUN_BEDROCK=1 pytest -m bedrock` (needs AWS creds)
- Cloud handler: `tests/integration/test_runtime.py`; cloud units: `tests/unit/test_cloud.py`
- Current status: **61 passed, 1 deselected** (offline) + **1 passed** (live Bedrock), ruff clean
