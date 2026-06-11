# 15-Minute Talk Outline - Agentic Quantum Error Mitigation Lab

Talk goal: explain what this project builds, why it matters, how the adaptive loop works, and what the demo proves.

Suggested format: 10 slides, about 90 seconds each.

---

## Slide 1 - Agentic Quantum Error Mitigation Lab

Slide content

- VLM-guided, adaptive quantum error mitigation on Amazon Braket
- Built on Mitiq techniques: REM, ZNE, and Pauli Twirling
- Deterministic DAG orchestration with policy-gated side effects
- Goal: hit a target accuracy with fewer shots than a fixed full-stack baseline

Visual / layout

- Center: title and one-sentence thesis
- Bottom: small pipeline strip: Probe -> Decide -> Mitigate -> Validate -> Report
- Right side callout: "Adaptive: 36k shots, error 0.0246; baseline: 84k shots, error 0.0870"

Image prompt

A clean technical title slide for an AWS quantum computing project. Show a simplified quantum circuit flowing into a decision loop, then into a chart labeled accuracy vs shots. Use a professional cloud architecture style, white background, dark text, AWS orange accent, quantum blue accent, no people.

---

## Slide 2 - The Problem: QEM Is Powerful, but Often Run Blindly

Slide content

- Today's noisy quantum programs need mitigation before results are useful
- Mitiq provides strong tools: readout error mitigation, zero-noise extrapolation, Pauli twirling
- The common workflow is static: choose a full recipe, set shot budgets, run everything
- Static full-stack mitigation can waste shots on techniques that do not help this run
- The project asks: can the system observe first, adapt second, and stop early?

Visual / layout

- Left: "Blind full stack" path: REM + PT + ZNE, all runs, fixed shots
- Right: "Adaptive loop" path: probe, select minimum technique set, retry only if needed
- Bottom takeaway: "QEM should be target-driven, not recipe-driven."

Image prompt

A two-column technical comparison diagram. Left column shows a heavy fixed pipeline with three stacked modules labeled REM, PT, ZNE and many small shot counters. Right column shows a compact feedback loop labeled probe, strategy, execute, validate, early stop. Clean lines, modern research presentation style.

---

## Slide 3 - Core Idea: Deterministic Agent, Not Free-Form Automation

Slide content

- The "agent" is a deterministic DAG with a small number of controlled decisions
- AI is used where visual judgment helps: probe histograms and ZNE plots
- Numeric rules are always the fallback and run first
- Every external action is checked by Policy before execution
- The result is adaptive, but bounded, reproducible, and auditable

Visual / layout

- Main diagram: deterministic rails with optional VLM sidecar
- Three labels: Rules floor, VLM judgment, Policy gate
- Show VLM arrows into `empirical_probe` and `validate`

Image prompt

A precise systems diagram showing a DAG pipeline running on rails. A side module labeled VLM inspects plots and returns structured JSON. A gate labeled Policy sits before tool execution. The style is restrained, engineering-focused, with small labels and clear arrows.

---

## Slide 4 - Architecture: Local-First Core, Cloud-Ready Wrapper

Slide content

- Local core runs entirely on Braket `LocalSimulator` with named noise models
- `src/aqem/braket_mitiq`: vendored Braket + Mitiq execution primitives
- `src/aqem/dag` and `src/aqem/nodes`: orchestration and QEM stages
- `src/aqem/vlm` and `src/aqem/tools`: Bedrock Claude or SageMaker-compatible VLM seam
- `src/aqem/cloud`: AgentCore Runtime handler, S3/local artifacts, optional Guardrails

Visual / layout

- Layered architecture:
  - CLI / AgentCore Runtime
  - DAG engine + nodes
  - Policy + audit
  - Braket/Mitiq tool and VLM tool
  - Local simulator now, cloud runtime wrapper ready

Image prompt

A layered software architecture diagram for a quantum error mitigation agent. Top layer CLI and AgentCore Runtime, middle layer DAG engine and policy audit, lower layer Braket/Mitiq tool and VLM plot-inspection tool, base layer LocalSimulator and optional AWS services. Use compact AWS-style boxes.

---

## Slide 5 - The Eight-Node QEM DAG

Slide content

- `empirical_probe`: run cheap readout and GHZ probes
- `strategy_select`: choose minimal mitigation strategy from probe classification
- `readout_calibrate`: build REM inverse confusion matrix
- `circuit_generate` and `execute`: build variants and run Program Sets
- `post_process`: apply REM/PT/ZNE pipeline and compute estimate
- `validate`: stop, retry shots, recalibrate, or escalate strategy
- `report`: emit estimate, shots used, audit, and baseline comparison

Visual / layout

- Horizontal DAG:
  `empirical_probe -> strategy_select -> readout_calibrate -> circuit_generate -> execute -> post_process -> validate -> report`
- Retry arrows from `validate` back to `execute`, `readout_calibrate`, or `strategy_select`

Image prompt

A horizontal DAG workflow diagram with eight labeled nodes for a quantum error mitigation pipeline. Include colored retry arrows from validate back to execute, readout_calibrate, and strategy_select. White background, technical labels, no decorative clutter.

---

## Slide 6 - Decision Logic: Probe Cheaply, Escalate Only When Needed

Slide content

- Probe classification: readout, gate/coherent, or shot-noise dominated
- Readout-dominated runs start with REM only
- Gate/coherent runs add ZNE; PT is added later if needed
- Validation uses true error in local benchmark mode, otherwise error bar
- Retry policy:
  - close to target -> more shots
  - far from target -> stronger strategy
  - readout anomaly -> recalibrate
  - target met -> stop immediately

Visual / layout

- Decision tree with four leaves: STOP, RETRY_SHOTS, RETRY_CALIBRATION, RETRY_STRATEGY
- Small side box: "Minimum sufficient mitigation bias"

Image prompt

A decision tree for adaptive quantum error mitigation. Start with probe classification, branch to REM only, REM plus ZNE, or more shots, then validate against target. Leaves are stop, more shots, recalibrate, escalate strategy. Clean dark text with subtle green, amber, and red status colors.

---

## Slide 7 - Policy and Audit: Safety Is Deterministic

Slide content

- Controlled action set only: run REM, ZNE sweep, PT, increase shots, recalibrate REM, stop and report
- Budget hard gate rejects actions that exceed remaining shot or cost budget
- No-device-recalibration guard blocks vendor calibration or device-metadata actions
- Retry cap prevents loops from running forever
- Every approved and rejected action is appended to an audit record

Visual / layout

- Gate diagram: Agent proposal -> Policy checks -> Braket/Mitiq execution
- Audit log shown as a side ledger
- Include the phrase: "AI proposes; Policy disposes."

Image prompt

A security and governance diagram for a scientific agent. Show an action request entering a policy gate with checks for allowed action, budget, no device recalibration, and retry cap. Approved actions go to execution, rejected actions go to an audit log. Minimal professional style.

---

## Slide 8 - Demo Setup: 2-Qubit Ising Estimate on a Noisy Simulator

Slide content

- Task: estimate the transverse-field Ising Hamiltonian expectation value `<H>`
- Device: Braket LocalSimulator with `qd_readout_2` readout-dominated noise
- Baseline: fixed full stack with REM + PT + ZNE
- Adaptive: probe -> classify readout-dominated -> REM-only -> validate -> stop
- CLI entry point: `aqem report --device qd_readout_2 --target 0.06 --seed 7`

Visual / layout

- Left: simple 2-qubit ansatz circuit
- Middle: noise model badge `qd_readout_2`
- Right: command output table placeholder

Image prompt

A technical demo slide showing a small two-qubit circuit, a simulator/noise-model block labeled qd_readout_2, and an output table comparing baseline and adaptive. Make it look like a polished developer demo, with monospace command text and compact visual hierarchy.

---

## Slide 9 - Result: Same Target, Fewer Shots

Slide content

- Ideal reference: 1.8041, target accuracy: 0.06
- Static baseline: 84,000 shots, error 0.0870, techniques REM + PT + ZNE
- Adaptive loop: 36,000 shots, error 0.0246, technique REM
- Shots saved: 48,000
- Shot ratio: 2.33x
- Efficiency gain demonstrated: adaptive hit target with fewer shots and better accuracy

Visual / layout

- Main chart: bar chart for shots, baseline vs adaptive
- Secondary chart or annotation: error relative to target line
- Big takeaway: "The loop avoided unnecessary ZNE/PT work."

Image prompt

A clean result slide with two charts. First chart compares shots: baseline 84k and adaptive 36k. Second chart compares absolute error against a horizontal target line at 0.06: baseline 0.087 and adaptive 0.0246. Professional scientific presentation style, clear labels.

---

## Slide 10 - What This Enables Next

Slide content

- A reusable pattern for target-driven QEM experiments
- Local-first development with cloud runtime seams
- Bedrock Claude VLM is available today; SageMaker-hosted Ising VLM remains a provider swap
- AgentCore Runtime wrapper, artifact storage, and Guardrails are implemented for cloud packaging
- Future work: live Braket QPUs, richer characterization, neural QEC decoding
- Closing thesis: governed adaptivity can make quantum experiments cheaper and more interpretable

Visual / layout

- Roadmap with three lanes:
  - Now: local simulator demo and CLI
  - Cloud packaging: AgentCore Runtime, artifacts, Guardrails
  - Next: QPU execution and QEC decoding
- End with one sentence: "Observe cheaply. Mitigate adaptively. Stop when enough."

Image prompt

A concise roadmap slide for a quantum software project. Three horizontal lanes labeled now, cloud packaging, and next. Include icons for simulator, cloud runtime, S3 artifact, guardrail shield, QPU, and decoder. Clean AWS-inspired style with restrained colors.
