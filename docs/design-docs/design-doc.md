# Agentic Quantum Error Mitigation Lab

> *VLM-guided, adaptive quantum error mitigation (Mitiq) on Amazon Braket, orchestrated by Amazon Bedrock AgentCore.*

---

## 1. Overview

### 1.1 Summary

A **cloud-native agentic loop that makes quantum error mitigation (QEM) more efficient**. An **Amazon Bedrock AgentCore** agent borrows NVIDIA Ising Calibration's core idea — a VLM that inspects experiment plots and drives a DAG workflow with adaptive retry — and applies it to **Mitiq-based error mitigation** on Amazon Braket. Instead of running a fixed mitigation recipe blindly, the agent selects techniques, tunes their parameters, and stops early once a target accuracy is reached — reaching that accuracy with **fewer shots and lower cost**.

### 1.2 Core Thesis

Mitiq provides strong QEM techniques (ZNE, PT, REM, and more), but in practice they are applied **statically**: a fixed technique stack with hand-picked parameters (ZNE scale factors, twirl counts, shot budgets), run as sequential notebook cells with no feedback. This is wasteful — shot overhead is spent on techniques that don't help, parameters that are mis-tuned, or runs that were already good enough.

NVIDIA Ising Calibration showed a better pattern for a *different* problem (lab QPU calibration): a **VLM inspects plots and guides a DAG workflow with adaptive retry**. We transplant that pattern onto QEM:

- **Characterize cheaply, then decide** — run small probe circuits, let the agent (rules + VLM) pick the mitigation strategy from what it observes.
- **Inspect intermediate plots with a VLM** — judge whether a ZNE extrapolation is physically reasonable or whether a result is already within target before spending more shots.
- **Adapt and retry only when needed** — escalate parameters (more twirls, different ZNE factory, more shots) only when quality is insufficient; stop early when it is.

> **Efficiency thesis: VLM-guided adaptive QEM reaches a target accuracy with materially fewer shots/cost than running the full Mitiq stack blindly.**

> **AI does not replace the quantum expert; it accelerates the observe → diagnose → mitigate → decide loop — as a governed cloud agent.**

### 1.3 Scope & Constraint

We **cannot access device-level details** (vendor calibration metadata, per-qubit T1/T2/fidelities, topology) right now. The design therefore relies on **empirical characterization** — cheap probe circuits run on Braket **local simulators and noise models** (with the local device emulator and real QPUs as a later step) — not on device-property queries. **QEC decoding is explicitly future work** (§9); the focus here is QEM efficiency.

### 1.4 Sources and Services

**Reused IP** — NVIDIA [Ising](https://github.com/NVIDIA/Ising) (VLM model family) and the [Quantum-Calibration-Agent-Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint) (DAG workflow logic, VLM prompt patterns); [Mitiq](https://github.com/unitaryfund/mitiq) (ZNE/PT/REM/… techniques); Amazon Braket [examples](https://github.com/amazon-braket/amazon-braket-examples) and [SDK](https://github.com/amazon-braket/amazon-braket-sdk-python) (Program Sets, simulators, noise models).

**AWS services that run the system:**

| Service | Role |
|---|---|
| Bedrock AgentCore **Runtime** | Serverless host for the agent loop + DAG orchestration |
| Bedrock AgentCore **Gateway** | Exposes Braket/Mitiq and the VLM as MCP tools; routes calls. *As built:* a FastMCP server (`aqem.cloud.mcp_server`) deployed as a second MCP-protocol Runtime; the loop routes to it (SigV4) when `AQEM_TOOL_TRANSPORT=mcp`, else calls the same tool functions in-process (the default). |
| custom **Policy** (+ Bedrock **Guardrails**) | Deterministic safety/cost audit of every action. *As built:* the authoritative layer is an in-process deterministic validator (`aqem.policy`) running inside the Runtime; Bedrock Guardrails is an optional managed content-safety check on free-text I/O. |
| AgentCore **Identity / Memory / Observability** | Scoped IAM, session + history context, traces/audit |
| Amazon **Bedrock** (Claude VLM) | Managed Claude (Sonnet 4.5) inspects plots and drives PASS/retry decisions (Tool 1). *(The original SageMaker-hosted Ising VLM is dropped from scope — see §9.)* |
| Amazon **Braket** | Simulators, noise models, Program Sets; QPUs later (Tool 2) |
| Amazon **S3** | Large artifacts: arrays, plots, audit trail |

**Strategy — port the *idea*, not the lab pipeline.** Reuse the blueprint's DAG engine and VLM prompt patterns, but apply them to QEM rather than QPU calibration; run them on AgentCore Runtime, expose tools through Gateway, and host the VLM on SageMaker.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. A session-ready demo showing VLM-guided adaptive QEM that hits a target accuracy with fewer shots than a fixed full-stack Mitiq baseline.
2. Reuse the Ising Calibration *pattern* (VLM + DAG + adaptive retry) on AgentCore Runtime.
3. Use Mitiq techniques (ZNE, PT, REM) on Braket simulators/noise models, invoked via Gateway.
4. Use the Ising Calibration VLM (on SageMaker) to inspect mitigation plots and drive PASS / retry decisions.
5. Govern every action through a deterministic Policy/Guardrails layer (controlled action set + cost hard gates) with full audit.
6. Report an explicit **efficiency comparison** (accuracy vs shots/cost) against the static baseline.

### 2.2 Non-Goals

- **No device-detail dependence.** The agent does not query vendor calibration metadata; characterization is empirical. It performs **mitigation-strategy diagnosis**, not vendor QPU recalibration.
- **No QEC decoding in scope** — neural decoding is future work (§9).
- **No bespoke agent infra** — hosting, tool routing, identity, memory, observability are delegated to AgentCore.

---

## 3. The Ising Calibration Pattern, Applied to QEM

| Ising Calibration (lab QPU) | This design (Mitiq QEM) |
|---|---|
| DAG nodes (spectroscopy, rabi, t1, ramsey) | QEM stages (probe, strategy, calibrate, execute, post-process, validate) |
| Node dependencies (topological order) | Stage ordering (REM calibration before execution; folding before twirling) |
| VLM inspects calibration curves | VLM inspects ZNE extrapolation / readout-distribution plots |
| Adaptive retry (wider sweep, more averages) | Adaptive retry (more twirls, different ZNE factory, more shots) |
| Decide next experiment | Decide next mitigation action (or stop early) |
| Querying lab instruments | **Empirical probe circuits** (no device metadata) |
| Goal: a calibrated qubit | Goal: a target-accuracy estimate at **minimum shot cost** |

---

## 4. Cloud-Native Architecture

### 4.1 Component & Loop View

```
                    Researcher
                       │  (1) estimation task + target accuracy + budget
                       v
┌──────────────── AWS Cloud — Region ──────────────────────────────────┐
│  ┌──────────── Amazon Bedrock — AgentCore ─────────────────────────┐  │
│  │   [Runtime] ──(2) actions──> [Policy] ──(3) approved──> [Gateway]│  │
│  │   orchestration              safety & audit            tool route│  │
│  │      ^                                                  │     │   │  │
│  │      │ (6) results / plots — close the loop             │     │   │  │
│  └──────┼──────────────────────────────────────────────────┼─────┼───┘  │
│         │                              (4) plot analysis     v     v (5)  │
│         │                                          ┌─────────┐ ┌────────┐ │
│         └──────────────────────────────────────────│SageMaker│ │ Braket │ │
│                                                     │Ising VLM│ │ + Mitiq│ │
│                                                     └─────────┘ └────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

**Closed loop:** (1) researcher submits an estimation task with a target accuracy and shot/cost budget to **Runtime**; (2) Runtime drives the DAG and proposes actions reviewed by **Policy**; (3) approved actions go to **Gateway**; (4) Gateway routes plot analysis to the **SageMaker** Ising VLM (Tool 1); (5) and circuit execution / Mitiq mitigation to **Braket** (Tool 2); (6) results and plots return to Runtime, which decides whether to stop or adapt.

### 4.2 Role of Each Primitive

- **Runtime** — serverless host for the agent loop and DAG state machine; tolerates long simulator/queue waits; per-session isolation. All side effects pass Policy first, then run either in-process or via the Gateway.
- **Policy** — deterministic layer between "agent proposes" and "tool executes." *As built,* this is an in-process validator (`aqem.policy`) running inside the Runtime container — not a managed AWS service — enforcing the controlled action set (§6.3), the cost/shot budget hard gates (§7), and the no-device-recalibration guard (§2.2). **Bedrock Guardrails** sits alongside it as an optional managed content-safety check on free-text I/O; the deterministic Policy is authoritative. Every decision is audited (append-only JSONL, persisted to S3) via Observability. The node → `Policy.check()` → tool order means the Gateway never executes an ungated action.
- **Gateway** — exposes two MCP tool groups — **Tool 1** (the Claude VLM, plot diagnosis) and **Tool 2** (Braket execution + Mitiq mitigation) — so DAG nodes call tools by name. *As built,* the tools live behind a `ToolClient` seam (`aqem.tools.client`): in-process by default (identical numerics, offline, no AWS), or routed over MCP to a FastMCP server (`aqem.cloud.mcp_server`) when `AQEM_TOOL_TRANSPORT=mcp`. That server is deployed as its own **MCP-protocol AgentCore Runtime** and invoked SigV4-signed at the `bedrock-agentcore` data plane; the server reconstructs the serialized arguments (circuit ↔ OpenQASM, device-by-name, calibration ↔ nested-list matrix) and calls the same tool functions, so the physics is transport-independent.
- **Bedrock (Claude VLM)** — managed Claude (Sonnet 4.5, `ChatBedrockConverse`) inspects the rendered plots; when the Gateway is in use the VLM runs server-side on the MCP runtime. *(The SageMaker-hosted Ising VLM is dropped from scope — §9.)*
- **Braket + Mitiq** — simulators and noise models for execution; Program Sets to batch circuit variants cheaply; Mitiq for ZNE/PT/REM.
- **Identity / Memory / Observability** — scoped IAM (no static creds), session + long-term context (large artifacts in S3), and end-to-end traces of nodes, tool calls, shots/costs, and Policy decisions.

---

## 5. Workflow (DAG in Runtime)

The DAG is the orchestration state machine in Runtime. Each node's external effects are Gateway tool calls that pass the Policy gate first. The loop is built to **spend the minimum shots needed to hit the target**.

```
[empirical_probe] ───────────┐
                            ├──> [strategy_select] ──> [readout_calibrate]
[problem_define] ────────────┘            │                    │
[cost_budget] ───────────────────────────┴──> [circuit_generate]
                                                     │
                                                [execute]
                                                     │
                                            [post_process]
                                                     │
                                             [validate] ── insufficient ──> adapt & retry upstream
                                                     │ target met
                                                  [report]  (efficiency vs baseline)
```

| Node | Purpose |
|---|---|
| `empirical_probe` | Run a few cheap diagnostic circuits (readout calibration, a Bell/GHZ probe) to estimate noise empirically — **no device metadata**. VLM inspects the probe histograms to flag the dominant error type (readout vs gate/coherent vs shot noise). |
| `strategy_select` (AI) | From probe results + target accuracy + budget, choose Mitiq techniques (REM/PT/ZNE/composite) and initial parameters (ZNE scale factors, twirl count, shot allocation). Start minimal; escalate only if needed. |
| `cost_budget` | Define shot/cost budget; feeds the Policy cost gate. |
| `readout_calibrate` | Build the REM inverse confusion matrix via readout twirling (Program Sets); adaptive retry on poor quality (more twirls → alternative qubit subset → stop). |
| `circuit_generate` | Build the ZNE-scaled × Pauli-twirl × readout-twirl circuit variants with bit masks. |
| `execute` | Batch variants into Program Sets (≤100/set) and run on a simulator/noise model (cost-gated); ~80–90× fewer task submissions vs individual tasks. |
| `post_process` | Mitiq pipeline: undo readout twirl → REM correction → expectation values → average over twirls → ZNE extrapolation; emit estimate + error bar. |
| `validate` (AI) | Compare error bar to target; VLM checks the extrapolation plot for anomalies (non-monotone decay, outliers). Decide STOP (target met), RETRY_SHOTS, RETRY_CALIBRATION, or RETRY_STRATEGY — **early-stop whenever the target is already met** to save shots. |
| `report` | Final estimate plus the **efficiency comparison**: accuracy and shots/cost vs the static full-stack baseline. |

---

## 6. AI Integration & Decision Logic

### 6.1 VLM Integration Points

All VLM calls route through Gateway to the SageMaker Ising VLM (Tool 1):

- **`empirical_probe`** — inspect probe histograms; classify the dominant error type and suggest which mitigation matters most.
- **`validate`** — judge whether the ZNE extrapolation is physically reasonable (monotone decay, no wild outliers), whether the readout distribution is anomalous, and whether the improvement is meaningful vs shot noise; recommend STOP or a specific retry mode. Returns structured JSON.

The blueprint's prompt patterns are reused; outputs are always structured JSON for deterministic downstream handling.

### 6.2 Hybrid Decision Strategy

Deterministic rules run first and are the fallback when VLM output is uncertain; the VLM adds visual judgment and explanation. The bias is always toward **minimum sufficient mitigation**: begin with the cheapest technique implied by the probe (often REM), add ZNE/PT only when the validate step shows the target is not met, and stop as soon as it is.

### 6.3 Controlled Action Set (enforced by Policy)

The agent may only request these; Policy rejects anything else:

```
increase_shots · run_readout_confusion_matrix · run_readout_mitigation ·
run_zne_sweep · run_pauli_twirling · change_zne_factory · reduce_technique_set ·
stop_and_report
```

### 6.4 Adaptive Retry & Cascading Invalidation

Retry maps the Ising-calibration pattern to QEM actions: error bar close to target → more shots; readout drift → recalibrate; fit anomaly → change ZNE factory (Linear → Exponential → Richardson); negligible improvement → reduce/switch technique set; target met → stop.

Unlike single-node lab retry, this needs **DAG-aware invalidation** (Runtime logic): a `readout_calibrate` retry invalidates `circuit_generate`/`execute`/`post_process`; a `strategy_select` change invalidates the whole downstream chain; tracked via an `invalidated_by` field per node.

---

## 7. Cost & Safety

- **Policy cost gate** — every execute/mitigate action carries a predicted shot/cost estimate; Policy refuses to approve anything exceeding the remaining budget. Because the loop early-stops and starts minimal, total shots are bounded by what's needed for the target — the central efficiency mechanism.
- **Program Sets** — always batch circuit variants through them (≤100/set) to minimize per-task overhead.
- **Agent infra** — Runtime is serverless (no idle cost); the SageMaker VLM endpoint is the main standing cost, so use serverless/async inference or deploy on-demand and tear down after the session; Gateway/Identity/Memory/Observability costs are minor.

---

## 8. Demo

**Prompt:** *"Estimate ⟨observable⟩ for this circuit on a noisy Braket simulator to within target precision, using as few shots as possible. Mitigate errors and show me why."*

**Flow:** researcher submits to Runtime → empirical probe + VLM error-type classification → strategy starts minimal (e.g. REM only) → Policy-approved execution via Program Sets → Mitiq post-processing → VLM inspects the result/extrapolation → if target met, **stop**; otherwise adapt (add ZNE / more twirls / more shots) and retry → final report with the **shots-vs-accuracy comparison against the static full-stack baseline**, all traced in Observability.

**Headline result:** same target accuracy, materially fewer shots/cost than running the full Mitiq stack blindly — with every action audited.

> **This demo turns Amazon Braket + Mitiq into a governed, agentic QEM loop on Amazon Bedrock AgentCore: probe cheaply, mitigate adaptively under an Ising-Calibration-style VLM, stop early, and prove the efficiency gain.**

---

## 9. Future Work

- **Neural QEC decoding** — extend the loop with NVIDIA [Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding) (3D CNN Fast/Accurate pre-decoders + PyMatching baseline) on syndrome data, bridging circuit-level mitigation to logical-level error correction. Out of scope for now.
- **Real device execution** — once device access is available, add the Braket local device emulator and QPUs as execution targets; the empirical-probe design carries over unchanged (it never depended on device metadata).
- **Richer characterization** — optionally incorporate device calibration metadata when accessible, as an *additional* signal to `strategy_select`.

---

## 10. Key Design Decisions

| Aspect | Ising Calibration (Lab) | This design (QEM on AgentCore) | Decision |
|---|---|---|---|
| Problem | QPU calibration | Mitiq error-mitigation efficiency | Transplant the VLM+DAG+retry pattern |
| Characterization | Lab instrument queries | **Empirical probe circuits** | No device-metadata dependence |
| Hosting | Local CLI / subprocess | Runtime (serverless) | Port DAG engine to a Runtime entrypoint |
| Tool invocation | In-process calls | In-process (default) **or** Gateway/MCP (opt-in) | `ToolClient` seam; MCP server live as a 2nd Runtime, routed via `AQEM_TOOL_TRANSPORT=mcp` |
| VLM hosting | NVIDIA NIM / Build API | Managed Claude on Bedrock | Use Bedrock Claude (Sonnet 4.5); SageMaker Ising VLM dropped (§9) |
| Safety / audit | Manual / pause-and-suggest | In-process Policy (+ optional Guardrails) + cost gate | Deterministic pre-dispatch review inside the Runtime |
| Efficiency | n/a | Early-stop + minimal-sufficient mitigation | Budget-gated, target-driven loop |
| Decoding | n/a | Future work | Excluded from current scope |

---

## 11. Milestones

| Phase | Outcome |
|---|---|
| **0 — Cloud setup** | AgentCore Runtime/Gateway/Identity skeleton; Ising VLM on SageMaker; Braket SDK + Mitiq; agent can call the VLM via Gateway. |
| **1 — Empirical probe + baseline** | Probe circuits + a static full-stack Mitiq baseline on a noisy simulator, with a shots/accuracy measurement harness. |
| **2 — VLM-guided strategy** | VLM error-type classification and validate-step plot inspection integrated; agent picks and early-stops mitigation. |
| **3 — Adaptive loop + Policy** | Full DAG on Runtime with adaptive retry, Policy guardrails + cost gate; **efficiency comparison vs baseline** demonstrated. |
| **4 — Future** | Real-device execution; neural QEC decoding (§9). |

---

## 12. Risks

| Risk | Impact | Mitigation |
|---|---:|---|
| No device details / no live QPU | — | Design targets simulators + noise models; empirical probe needs no metadata |
| Efficiency gain not significant | High | Pick observables/noise where blind full-stack mitigation is clearly wasteful; report honest shots-vs-accuracy curves |
| VLM output variance | Medium | Deterministic rules first, VLM for judgment/explanation |
| Overclaiming QPU calibration | Medium | Policy blocks recalibration actions; "mitigation diagnosis" wording |
| Cost overrun | Medium | Policy cost gates; Program Sets; tear down SageMaker endpoint after demo |
| AgentCore/SageMaker region availability | Medium | Verify in the demo region during Phase 0 |

---

## References

1. NVIDIA Developer Blog — [Ising: AI-Powered Workflows for Fault-Tolerant Quantum Systems](https://developer.nvidia.com/blog/nvidia-ising-introduces-ai-powered-workflows-to-build-fault-tolerant-quantum-systems/)
2. AWS — [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)
3. AWS Blog — [Error mitigation on Braket with program sets and Mitiq](https://aws.amazon.com/blogs/quantum-computing/error-mitigation-on-amazon-braket-with-program-sets-and-mitiq/)
4. [Mitiq](https://github.com/unitaryfund/mitiq) · GitHub — [NVIDIA/Ising](https://github.com/NVIDIA/Ising) · [Quantum-Calibration-Agent-Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint) · [Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding) (future work)
5. Takagi et al. (2022); Wallman & Emerson (2015); Bravyi et al. (2021)
