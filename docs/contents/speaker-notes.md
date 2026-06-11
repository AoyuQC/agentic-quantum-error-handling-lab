# Speaker Notes - 15-Minute Talk

Talk title: Agentic Quantum Error Mitigation Lab

Timing guide: 10 slides, about 90 seconds each.

---

## Slide 1 - Agentic Quantum Error Mitigation Lab

Hi everyone, I am Aoyu. In the next fifteen minutes I want to walk through this project: an Agentic Quantum Error Mitigation Lab.

The short version is this. We are taking quantum error mitigation techniques from Mitiq, running them on Amazon Braket, and putting an adaptive decision loop around them. The goal is not to make the agent mysterious or autonomous in an open-ended way. The goal is very practical: spend fewer shots to reach the same target accuracy.

The core idea is: probe cheaply, choose the minimum mitigation strategy that looks sufficient, validate the result, and stop as soon as the target is met. If the target is not met, the loop retries in a controlled way: more shots, recalibration, or a stronger mitigation strategy.

The headline demo is a two-qubit Ising expectation estimate. A blind full-stack baseline uses 84,000 shots and still misses the target in this readout-dominated setting. The adaptive loop uses 36,000 shots, applies only readout error mitigation, and reaches better accuracy. That is a 2.33x shot reduction in the demo.

So the thesis is: QEM should be target-driven, not recipe-driven.

---

## Slide 2 - The Problem: QEM Is Powerful, but Often Run Blindly

Quantum error mitigation is essential because current devices are noisy. We do not have full fault tolerance yet, so if we want useful estimates from near-term circuits, we need mitigation.

Mitiq gives us a strong toolbox. Readout error mitigation helps when measurement error dominates. Zero-noise extrapolation helps reason about gate noise by running noise-scaled versions of the circuit. Pauli twirling can turn coherent errors into a more stochastic form that is easier to average.

The issue is not that these techniques are weak. The issue is that they are often used statically. You pick a full stack, maybe REM plus PT plus ZNE, choose scale factors, twirl counts, and shot budgets, then run everything. That is fine as a baseline, but it can be wasteful.

If the dominant problem is readout error, then running a large ZNE and Pauli-twirling stack may spend shots without improving the final estimate. Worse, the extra processing can sometimes add variance or bias. In this project, that is exactly the inefficiency we expose.

So we ask a simple systems question: can the workflow observe the noise first, select the smallest reasonable strategy, and only escalate when validation says escalation is needed?

---

## Slide 3 - Core Idea: Deterministic Agent, Not Free-Form Automation

When I say "agentic" here, I do not mean an LLM randomly deciding what quantum experiment to run next. The project is intentionally more disciplined than that.

The backbone is a deterministic DAG engine. The nodes are fixed. The retry paths are fixed. The possible actions are fixed. The agent adapts, but only inside this controlled state machine.

The VLM is used in two places where visual judgment is helpful. First, during empirical probing, it can inspect readout and GHZ histograms and classify the dominant error source. Second, during validation, it can inspect a ZNE plot and flag things like non-monotone extrapolation, outliers, or readout anomalies.

But numeric rules always exist underneath. If there is no VLM client, or the VLM output is low confidence, malformed, or unavailable, the loop falls back to deterministic rules. That is important for reproducibility and for testing.

And before anything side-effecting runs, a deterministic Policy checks it. So the shape is: rules provide the floor, VLM adds judgment, and Policy is the hard gate.

---

## Slide 4 - Architecture: Local-First Core, Cloud-Ready Wrapper

The project is built local-first. The core loop runs on a Braket LocalSimulator with named noise models, so unit tests and integration tests do not need AWS infrastructure.

At the bottom, `src/aqem/braket_mitiq` vendors the Braket and Mitiq primitives used for mitigation. This includes Program Sets, readout twirling, observable grouping, and mitigation helpers.

Above that, `src/aqem/tools` exposes the Braket/Mitiq execution seam and the VLM plot-inspection seam. The DAG nodes call these tool-shaped functions, which keeps the orchestration separate from execution details.

The orchestration layer is `src/aqem/dag` and `src/aqem/nodes`. That is where the deterministic engine, topological ordering, retry loop, and cascading invalidation live.

The VLM layer is provider-pluggable. The default is managed Bedrock Claude, with a SageMaker provider present as a compatible swap for an Ising-style VLM endpoint.

Finally, there is a cloud wrapper in `src/aqem/cloud`. It provides an AgentCore Runtime-style handler, local or S3 artifact storage, and optional Bedrock Guardrails around free-text input and output. The tools are invoked in-process today, but the seams are clean for a Gateway-style split later.

---

## Slide 5 - The Eight-Node QEM DAG

Here is the actual workflow.

The first node is `empirical_probe`. It runs cheap readout and GHZ probe circuits, produces histograms, and classifies whether the dominant issue looks like readout error, gate or coherent error, or mostly shot noise.

Then `strategy_select` chooses a starting strategy. The important bias is minimum sufficient mitigation. For readout-dominated noise, it starts with REM only. For gate/coherent noise, it can include ZNE. PT is something the strategy can add later if needed.

`readout_calibrate` builds the REM inverse confusion matrix. Then `circuit_generate` prepares the measurement and mitigation variants, and `execute` runs them through Program Sets.

`post_process` applies the mitigation pipeline, computes the expectation estimate, attaches an error bar, and emits any plot data.

`validate` compares the metric against the target. In the local benchmark, we have an exact noiseless reference, so validation can use true absolute error. In settings without that reference, the error bar is the proxy.

Finally, `report` emits the final estimate, total shots, audit trail, and comparison data.

The key engineering detail is retry invalidation. If validation asks for more shots, only execution and downstream nodes need to rerun. If readout calibration is stale, calibration and downstream work rerun. If the strategy changes, the whole downstream chain is invalidated.

---

## Slide 6 - Decision Logic: Probe Cheaply, Escalate Only When Needed

The decision logic is deliberately small.

The probe step returns a dominant error class. If it is readout-dominated, the initial strategy is REM. If it is gate or coherent-error dominated, the initial strategy is REM plus ZNE. If the issue looks like shot noise, the loop leans toward more shots rather than stacking mitigation techniques.

During validation, the loop asks: did we meet the target? If yes, stop immediately. That early stop is the main efficiency mechanism.

If we missed the target but are close, the cheapest escalation is usually more shots. If we are far from target, the strategy escalates: add ZNE if absent, add PT if absent, or advance the ZNE factory. If the VLM confidently sees a readout anomaly, validation can trigger recalibration. If it sees a bad ZNE shape, it can recommend a strategy retry.

This is also why the project is easy to reason about. There are only four validation outcomes: stop, retry shots, retry calibration, or retry strategy.

So the agent is not inventing new science on the fly. It is automating the observe, diagnose, mitigate, and validate loop that a careful quantum engineer would otherwise run manually.

---

## Slide 7 - Policy and Audit: Safety Is Deterministic

The Policy layer is the part that keeps the adaptive loop governed.

First, there is a controlled action set. The agent can request actions like run readout mitigation, run a ZNE sweep, run Pauli twirling, increase shots, recalibrate the REM confusion matrix, reduce the technique set, or stop and report. Anything outside that set is rejected.

Second, there is a hard shot and cost budget. Each action carries a predicted shot count and optional cost. If the request would exceed the remaining budget, Policy rejects it before execution.

Third, there is a no-device-recalibration guard. This project is about mitigation diagnosis, not vendor QPU recalibration. Attempts to query device properties, pulse settings, T1 or T2 data, or calibration internals are blocked.

Fourth, retry caps prevent the loop from spinning indefinitely.

Every check is audited, approved or rejected. That means when the run finishes, we can explain not only the final estimate, but also which actions were requested, what they were expected to cost, and why they were allowed.

The short line is: AI proposes; Policy disposes.

---

## Slide 8 - Demo Setup: 2-Qubit Ising Estimate on a Noisy Simulator

The demo problem is intentionally small and reproducible.

We estimate the expectation value of a two-qubit transverse-field Ising Hamiltonian. The circuit is a shallow ansatz with Rx rotations and CZ gates. The observable is represented as weighted Pauli terms, and for local benchmarking the project computes an exact noiseless reference using linear algebra.

The noisy execution target is Braket LocalSimulator with the `qd_readout_2` noise model. This is a readout-dominated setting, which is useful because it lets us test whether the agent avoids over-mitigating.

The baseline is a blind full stack: readout error mitigation, Pauli twirling, and zero-noise extrapolation with a fixed budget. The adaptive loop starts by probing. The probe classification identifies readout as the dominant issue, so the strategy begins with REM only. Then validation checks whether that already hits the target.

The CLI command for the full comparison is:

`aqem report --device qd_readout_2 --target 0.06 --seed 7`

That runs both the static baseline and the adaptive loop, then prints the shots-vs-accuracy comparison.

---

## Slide 9 - Result: Same Target, Fewer Shots

Here is the result from the project README and integration acceptance test.

The ideal reference value is 1.8041, and the target absolute accuracy is 0.06.

The static baseline uses 84,000 shots with REM, PT, and ZNE. In this readout-dominated noise model, it ends with an error of 0.0870, so it does not meet the target.

The adaptive loop uses 36,000 shots. It chooses REM only, because the probe says the problem is readout-dominated. It ends with an error of 0.0246, which is comfortably inside the target.

So the adaptive loop saves 48,000 shots, or 2.33x fewer shots than the baseline. It is also more accurate in this demo.

The interpretation is not "REM is always better than full-stack mitigation." The interpretation is more useful: the right mitigation depends on the observed error mode. In this run, the full stack spends shots on ZNE and PT that are not needed. The adaptive loop avoids that work because it validates early.

That is the core win: not a new mitigation algorithm, but a better control loop around existing mitigation algorithms.

---

## Slide 10 - What This Enables Next

To close, this project gives us a reusable pattern for target-driven quantum experiments.

The local-first core makes it easy to test: unit tests cover decision rules, policy gates, VLM schema handling, efficiency metrics, DAG invalidation, and artifact handling. Integration tests run the local simulator path and the baseline-vs-adaptive demo.

The cloud wrapper points toward production use. The runtime handler accepts a payload, builds the problem and device, optionally constructs a Bedrock Claude VLM client, runs the adaptive loop, writes audit and comparison artifacts locally or to S3, and returns a compact run summary. Optional Bedrock Guardrails protect free-text input and output, while the deterministic Policy remains the authoritative execution gate.

There are three obvious next steps. First, run the same empirical-probe design against live Braket QPUs. Second, enrich characterization when device metadata is available, while keeping empirical probing as the portable baseline. Third, extend from error mitigation into neural QEC decoding on syndrome data.

The final takeaway is simple: observe cheaply, mitigate adaptively, and stop when enough. Governed adaptivity can make quantum experiments cheaper, easier to explain, and safer to automate.
