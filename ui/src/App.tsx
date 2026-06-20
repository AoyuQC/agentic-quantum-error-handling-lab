import { useCallback, useEffect, useState, type ReactNode } from "react";
import { fetchDevices, runStream } from "./api";
import { Chart } from "./Chart";
import type {
  Experiment,
  Figure,
  PlotRecord,
  ProgressEvent,
  RunRequest,
  RunResult,
  VlmTrace,
} from "./types";

type NodeState = "idle" | "running" | "done" | "failed";

type NodeEntry = {
  node: string;
  status: NodeState;
  shots?: number;
  detail?: Record<string, unknown>;
  plots?: PlotRecord[];
  // Live LLM output streamed while the node runs, accumulated per speaker
  // ("vlm" = image analysis tool, "agent" = orchestration decider).
  llm?: { vlm?: string; agent?: string };
};
type DecisionEntry = {
  action: string;
  reason: string;
  source?: string;
  metric_value?: number | null;
  target?: number;
};
type IterGroup = { iteration: number; nodes: NodeEntry[]; decision?: DecisionEntry };

// A ZNE noise-folded circuit variant (from circuit_generate): the base circuit
// re-expressed at an increasing noise scale (gate-folded, so deeper).
type FoldedCircuit = { scale: number; depth: number; n_gates: number; diagram: string };

// Display metadata for each pipeline node: number badge, icon, human label.
const NODE_META: Record<string, { n: string; icon: string; label: string }> = {
  empirical_probe: { n: "①", icon: "🔬", label: "Empirical probe" },
  strategy_select: { n: "②", icon: "🧩", label: "Strategy select" },
  readout_calibrate: { n: "③", icon: "🎛", label: "Readout calibrate" },
  circuit_generate: { n: "④", icon: "🛠", label: "Circuit generate" },
  execute: { n: "⑤", icon: "⚛", label: "Execute" },
  post_process: { n: "⑥", icon: "📊", label: "Post-process" },
  validate: { n: "⑦", icon: "✅", label: "Validate" },
  report: { n: "⑧", icon: "📄", label: "Report" },
};

// The per-iteration pipeline order (report runs once at the very end, so it is
// excluded here). Used to size the progress bar within an iteration.
const PIPELINE = [
  "empirical_probe",
  "strategy_select",
  "readout_calibrate",
  "circuit_generate",
  "execute",
  "post_process",
  "validate",
];

// Dr. Qubit's first-person narration per running step. Each node-id maps to a
// few phrasings; the live one is picked by iteration index (deterministic — no
// Math.random, which is unavailable here and would re-roll on every render).
// `**bold**` segments render as <strong> interjections (a tiny inline parser).
const NARRATION: Record<string, string[]> = {
  empirical_probe: [
    "**Aha** — fresh data! Let me peer into these probe plots and see what the qubits are hiding",
    "**Curious…** what are these qubits up to? Squinting at the probe histograms",
  ],
  strategy_select: [
    "**Now then…** which spell shall we cast? Picking the leanest mitigation recipe",
    "**Hmm** — minimal but sufficient. Choosing exactly the techniques this noise demands",
  ],
  readout_calibrate: [
    "**Steady hands** — calibrating the readout, inverting that pesky confusion matrix",
    "**Precision!** Mapping every bit-flip so I can undo the readout's little lies",
  ],
  circuit_generate: [
    "**Time to build!** Folding noise and twirling gates into a battalion of circuits",
    "**Onward** — compiling the mitigation circuits, scale by scale, twirl by twirl",
  ],
  execute: [
    "**Here we go** — firing the circuits at the device. Hold onto your wavefunction",
    "**Showtime!** Sampling shots from the noisy machine and collecting the counts",
  ],
  post_process: [
    "**The numbers are in!** Extrapolating to zero noise to squeeze out the true signal",
    "**Almost there** — folding the raw results into one clean estimate",
  ],
  validate: [
    "**Moment of truth** — is our error finally bowing to the target? Let me check",
    "**Eyes sharp** — comparing this estimate against the goal we set",
  ],
};
// VLM-on vision openers for the image-bearing steps.
const NARRATION_VLM: Record<string, string> = {
  empirical_probe: "Let me **analyze this image**… aha, the probe plots are telling a story",
  validate: "Let me **analyze this image**… does the extrapolation hold up to the target?",
};

// Pretty-print the observable as "1·ZI + 1·IZ + 0.5·XX".
function formatObservable(terms: [number, string][]): string {
  return terms
    .map(([c, p]) => `${Number.isInteger(c) ? c : c.toFixed(2)}·${p}`)
    .join(" + ");
}

function num(v: unknown, digits = 4): string {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

export default function App() {
  const [devices, setDevices] = useState<string[]>([]);
  const [req, setReq] = useState<RunRequest>({
    qubits: 2,
    target_accuracy: 0.06,
    device: "qd_readout_2",
    budget_shots: 2_000_000,
    use_vlm: false,
    compare_baseline: true,
    seed: 7,
  });
  const [running, setRunning] = useState(false);
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [timeline, setTimeline] = useState<IterGroup[]>([]);
  // Which iteration is expanded. null => follow the paced walkthrough (see
  // displayIter). A click on a rail chip pins one open.
  const [expandedIter, setExpandedIter] = useState<number | null>(null);
  // The iteration currently on stage in auto mode. Unlike the engine (which may
  // race ahead), this advances only after the shown iteration's step-by-step
  // walkthrough finishes AND a deliberate pause — so iterations don't flash by.
  const [displayIter, setDisplayIter] = useState<number | null>(null);
  // The iteration whose walkthrough has reached its terminal step (signalled up
  // from IterationStage). When it matches displayIter and a later iteration
  // exists, we hold ITER_PAUSE_MS then advance.
  const [settledIter, setSettledIter] = useState<number | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDevices()
      .then((d) => setDevices(d.devices))
      .catch(() => setDevices(["qd_readout_2", "qd_total"]));
  }, []);

  function set<K extends keyof RunRequest>(k: K, v: RunRequest[K]) {
    setReq((r) => ({ ...r, [k]: v }));
  }

  // Upsert a node into the timeline group for the given iteration (default: last).
  function upsertNode(iteration: number | undefined, entry: Partial<NodeEntry> & { node: string }) {
    setTimeline((groups) => {
      const next = groups.map((g) => ({ ...g, nodes: [...g.nodes] }));
      let gi = iteration != null ? next.findIndex((g) => g.iteration === iteration) : next.length - 1;
      if (gi < 0) {
        next.push({ iteration: iteration ?? next.length + 1, nodes: [] });
        gi = next.length - 1;
      }
      const nodes = next[gi].nodes;
      const ni = nodes.findIndex((n) => n.node === entry.node);
      if (ni >= 0) nodes[ni] = { ...nodes[ni], ...entry };
      else nodes.push({ status: "running", ...entry });
      return next;
    });
  }

  function onProgress(ev: ProgressEvent) {
    switch (ev.event) {
      case "experiment":
        setExperiment(ev as unknown as Experiment);
        break;
      case "iteration":
        setTimeline((g) => [...g, { iteration: ev.iteration!, nodes: [] }]);
        // Put the first iteration on stage immediately; later ones are pulled in
        // by the paced auto-advance (after the prior walkthrough + ITER_PAUSE_MS),
        // so the engine racing ahead doesn't skip the display past them.
        setDisplayIter((d) => (d == null ? ev.iteration! : d));
        break;
      case "node_start":
        if (ev.node) upsertNode(ev.iteration, { node: ev.node, status: "running" });
        break;
      case "node_done":
        if (ev.node)
          upsertNode(ev.iteration, {
            node: ev.node,
            status: "done",
            shots: ev.shots_used || undefined,
            detail: ev.detail,
            plots: ev.plots,
          });
        break;
      case "node_failed":
        if (ev.node) upsertNode(ev.iteration, { node: ev.node, status: "failed" });
        break;
      case "llm_delta":
        if (ev.node && ev.delta) {
          const role = (ev.role === "agent" ? "agent" : "vlm") as "vlm" | "agent";
          const piece = ev.delta;
          setTimeline((groups) => {
            const next = groups.map((g) => ({ ...g, nodes: [...g.nodes] }));
            let gi = ev.iteration != null ? next.findIndex((g) => g.iteration === ev.iteration) : next.length - 1;
            if (gi < 0) gi = next.length - 1;
            if (gi < 0) return groups;
            const nodes = next[gi].nodes;
            let ni = nodes.findIndex((n) => n.node === ev.node);
            if (ni < 0) {
              nodes.push({ node: ev.node!, status: "running" });
              ni = nodes.length - 1;
            }
            const prev = nodes[ni].llm || {};
            nodes[ni] = { ...nodes[ni], llm: { ...prev, [role]: (prev[role] || "") + piece } };
            return next;
          });
        }
        break;
      case "decision":
        setTimeline((groups) => {
          if (groups.length === 0) return groups;
          const next = [...groups];
          const gi = ev.iteration != null ? next.findIndex((g) => g.iteration === ev.iteration) : next.length - 1;
          const idx = gi >= 0 ? gi : next.length - 1;
          next[idx] = {
            ...next[idx],
            decision: {
              action: ev.action!,
              reason: ev.reason || "",
              source: ev.source,
              metric_value: ev.metric_value,
              target: ev.target,
            },
          };
          return next;
        });
        break;
    }
  }

  async function start() {
    setRunning(true);
    setResult(null);
    setError(null);
    setExperiment(null);
    setTimeline([]);
    setExpandedIter(null);
    setDisplayIter(null);
    setSettledIter(null);
    try {
      await runStream(req, { onProgress, onResult: setResult, onError: setError });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  // IterationStage signals up (via onSettled) when its walkthrough reaches the
  // terminal step. Memoised so the child effect doesn't re-fire on every render.
  const handleSettled = useCallback((iter: number) => setSettledIter(iter), []);

  // Paced auto-advance: once the iteration on stage has settled and a later
  // iteration exists, hold ITER_PAUSE_MS, then bring the next one on stage.
  useEffect(() => {
    if (expandedIter != null) return; // user pinned a specific iteration
    if (displayIter == null || settledIter !== displayIter) return;
    const hasNext = timeline.some((g) => g.iteration > displayIter);
    if (!hasNext) return;
    const t = setTimeout(() => {
      setDisplayIter((d) => {
        if (d == null) return d;
        const nexts = timeline.filter((g) => g.iteration > d).map((g) => g.iteration);
        return nexts.length ? Math.min(...nexts) : d;
      });
    }, ITER_PAUSE_MS);
    return () => clearTimeout(t);
  }, [expandedIter, displayIter, settledIter, timeline]);

  // Which iteration occupies the center stage: the user-pinned one, else the
  // paced walkthrough's current iteration (displayIter).
  const shownIter =
    expandedIter != null && timeline.some((g) => g.iteration === expandedIter)
      ? expandedIter
      : displayIter != null && timeline.some((g) => g.iteration === displayIter)
        ? displayIter
        : timeline.length
          ? timeline[timeline.length - 1].iteration
          : null;
  const shown = timeline.find((g) => g.iteration === shownIter) || null;

  return (
    <div className="console">
      {/* ---- side rail: settings · experiment · past iterations ---- */}
      <aside className="rail">
        <div className="rail-brand">
          <h1>Agentic QEM</h1>
          <p>An LLM agent that sees, reasons, and decides — like a quantum researcher.</p>
        </div>

        <div className="rail-panel">
          <h3>Run</h3>
          <label>Device
            <select value={req.device} onChange={(e) => set("device", e.target.value)} disabled={running}>
              {devices.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
          <div className="rail-row">
            <label>Target
              <input type="number" step={0.01} value={req.target_accuracy} disabled={running}
                onChange={(e) => set("target_accuracy", +e.target.value)} />
            </label>
            <label>Qubits
              <input type="number" min={1} max={6} value={req.qubits} disabled={running}
                onChange={(e) => set("qubits", +e.target.value)} />
            </label>
            <label>Seed
              <input type="number" value={req.seed ?? 7} disabled={running}
                onChange={(e) => set("seed", +e.target.value)} />
            </label>
          </div>
          <label className="tg">
            <input type="checkbox" checked={req.use_vlm} disabled={running}
              onChange={(e) => set("use_vlm", e.target.checked)} />
            Claude VLM
          </label>
          <label className="tg">
            <input type="checkbox" checked={req.compare_baseline} disabled={running}
              onChange={(e) => set("compare_baseline", e.target.checked)} />
            vs full-stack baseline
          </label>
          <button className="run" onClick={start} disabled={running}>
            {running ? "Running…" : "▶ Run agent loop"}
          </button>
        </div>

        {experiment && <ExperimentBanner exp={experiment} vlm={req.use_vlm} />}

        {timeline.length > 0 && (
          <div className="rail-panel">
            <h3>Iterations</h3>
            <div className="iter-list">
              {timeline.map((g) => (
                <button
                  key={g.iteration}
                  className={`iter-chip ${g.iteration === shownIter ? "active" : ""} ${
                    g.decision?.action === "stop" ? "met" : ""
                  }`}
                  onClick={() => setExpandedIter(g.iteration)}
                >
                  <span className="ic-n">#{g.iteration}</span>
                  <span className="ic-sum">{iterSummary(g)}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>

      {/* ---- center stage: the current iteration ---- */}
      <main className="stage">
        {error && <div className="panel" style={{ color: "var(--danger)" }}>Error: {error}</div>}

        {timeline.length === 0 && !error && (
          <div className="panel empty">
            Configure the run on the left and press <strong>Run agent loop</strong>. The active
            iteration will appear here — what the agent <strong>sees</strong>, how it{" "}
            <strong>reasons</strong>, and what it <strong>decides</strong>.
          </div>
        )}

        {timeline.length > 0 && (
          <ProgressBar timeline={timeline} running={running} done={!!result || !!error} />
        )}

        {timeline.length > 0 && (
          <Convergence
            timeline={timeline}
            target={experiment?.target_accuracy ?? req.target_accuracy}
          />
        )}

        {shown && (
          <IterationStage
            key={shown.iteration}
            group={shown}
            vlm={req.use_vlm}
            onSettled={handleSettled}
          />
        )}

        {result && <Results result={result} />}
      </main>
    </div>
  );
}

function ExperimentBanner({ exp, vlm }: { exp: Experiment; vlm: boolean }) {
  return (
    <div className="exp-banner">
      <div className="exp-title">
        <span className="exp-icon">🧪</span>
        <div>
          <div className="exp-name">{exp.description}</div>
          <div className="exp-sub mono">{formatObservable(exp.observable_terms)}</div>
        </div>
        <span className={`badge ${vlm ? "good" : ""}`} style={{ marginLeft: "auto" }}>
          {vlm ? "Claude VLM steering" : "rules-only"}
        </span>
      </div>
      <div className="exp-facts">
        <div><span className="k">Ansatz</span><span className="v">{exp.ansatz}</span></div>
        <div><span className="k">Ideal ⟨O⟩</span><span className="v mono">{exp.ideal.toFixed(4)}</span></div>
        <div><span className="k">Target</span><span className="v mono">≤ {exp.target_accuracy}</span></div>
        <div><span className="k">Device</span><span className="v mono">{exp.device}</span></div>
        <div><span className="k">Budget</span><span className="v mono">{exp.budget_shots.toLocaleString()} sh</span></div>
      </div>
    </div>
  );
}

// Determinate progress across all iterations: each iteration contributes the
// fraction of its pipeline nodes that have completed.
function ProgressBar({ timeline, running, done }: { timeline: IterGroup[]; running: boolean; done: boolean }) {
  const cur = timeline[timeline.length - 1];
  const stopped = timeline.some((g) => g.decision?.action === "stop");
  // Per-iteration completion is (# done nodes / pipeline length); the overall
  // bar shows the current iteration's progress (each iteration is a fresh pass).
  const doneNodes = cur ? cur.nodes.filter((n) => n.status === "done").length : 0;
  const frac = done || stopped ? 1 : Math.min(doneNodes / PIPELINE.length, 0.98);
  const runningNode = cur?.nodes.find((n) => n.status === "running");
  const label = done || stopped
    ? "complete"
    : runningNode
      ? `${NODE_META[runningNode.node]?.label || runningNode.node}…`
      : "starting…";
  return (
    <div className="progress">
      <div className="progress-meta">
        <span>Iteration {cur?.iteration ?? 1}</span>
        <span className="progress-node">{label}</span>
        <span className="progress-pct">{Math.round(frac * 100)}%</span>
      </div>
      <div className="progress-track">
        <div
          className={`progress-fill ${done || stopped ? "ok" : ""} ${running && !done ? "live" : ""}`}
          style={{ width: `${frac * 100}%` }}
        />
      </div>
    </div>
  );
}

// One-line summary shown when an iteration is folded.
function iterSummary(group: IterGroup): string {
  const strat = group.nodes.find((n) => n.node === "strategy_select");
  const techs = ((strat?.detail as any)?.strategy?.techniques as string[] | undefined) || [];
  const d = group.decision;
  const verdict = d
    ? d.action === "stop" ? "✓ target met" : `↻ ${d.action.replace("retry_", "retry ")}`
    : "running…";
  return `${techs.join(" + ") || "…"} — ${verdict}`;
}

// How long a finished step lingers before it fades, and the fade duration (ms).
// Together they pace the walkthrough so a human can follow one step at a time —
// this is an illustrative demo, so the display deliberately trails the engine.
const STEP_HOLD_MS = 3000;
const STEP_FADE_MS = 1400; // must match the .step-stage CSS transition
// A deliberate breather between iterations: once an iteration's walkthrough
// settles on its final step, hold here before the next iteration takes the
// stage, so the loop's iterations don't flash past.
const ITER_PAUSE_MS = 3000;

function IterationStage({
  group,
  vlm,
  onSettled,
}: {
  group: IterGroup;
  vlm: boolean;
  onSettled?: (iteration: number) => void;
}) {
  const met = group.decision?.action === "stop";
  const nodes = group.nodes;

  // Show ONE step at a time. The display walks the node list in order, pausing
  // on each finished step, fading it out, then fading the next one in — so it
  // trails the (often much faster) engine at a human-watchable pace. Each step
  // the walkthrough leaves behind drops into a folded list at the bottom.
  const [idx, setIdx] = useState(0);
  const [visible, setVisible] = useState(true);

  const current = nodes[idx];
  const hasNext = idx < nodes.length - 1;
  const finished = !!current && (current.status === "done" || current.status === "failed");

  useEffect(() => {
    // Advance only once the shown step is finished AND a next step has arrived.
    // Hold it briefly, fade out, then step forward and fade the next one in.
    if (!finished || !hasNext) return;
    const fadeOut = setTimeout(() => setVisible(false), STEP_HOLD_MS);
    const advance = setTimeout(() => {
      setIdx((i) => Math.min(i + 1, nodes.length - 1));
      setVisible(true);
    }, STEP_HOLD_MS + STEP_FADE_MS);
    return () => {
      clearTimeout(fadeOut);
      clearTimeout(advance);
    };
  }, [finished, hasNext, idx, nodes.length]);

  // Steps the walkthrough has already left behind, newest-first — folded at the
  // bottom, one click from full detail.
  const passed = nodes.slice(0, idx).reverse();
  // Locally track which passed steps the user re-expanded.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (node: string) =>
    setExpanded((s) => {
      const next = new Set(s);
      next.has(node) ? next.delete(node) : next.add(node);
      return next;
    });

  // The shown step is the terminal one once we've reached validate with a
  // decision and nothing follows — that's where the narrator settles.
  const settled = !hasNext && current?.node === "validate" && !!group.decision;

  // Tell the parent the walkthrough has reached its finished terminal step, so
  // it can hold the inter-iteration pause before advancing the stage.
  useEffect(() => {
    if (!hasNext && finished) onSettled?.(group.iteration);
  }, [hasNext, finished, group.iteration, onSettled]);
  const decisionFor = (n: NodeEntry) => (n.node === "validate" ? group.decision : undefined);

  // While `validate` runs it hasn't emitted its own plot yet, but `post_process`
  // just produced the same ZNE figure — fall back to it so the view isn't blank.
  const zneFallback = (
    nodes.find((n) => n.node === "post_process")?.plots?.find((p) => p.name === "zne_extrapolation")
      ?.data
  ) as Figure | undefined;
  // The ZNE folded mitigation set from circuit_generate — what differs between
  // iterations. Shown in the execute card (incl. while execute is still running).
  const folded = (
    nodes.find((n) => n.node === "circuit_generate")?.detail as any
  )?.folded_circuits as FoldedCircuit[] | undefined;

  return (
    <section className="iter-section open">
      <h2 className={`iter-title static ${met ? "met" : ""}`}>
        <span>▸ Iteration {group.iteration}</span>
        {group.decision && <span className="iter-summary">{iterSummary(group)}</span>}
      </h2>
      <div className="node-cards">
        {current && (
          <div className={`step-stage ${visible ? "step-in" : "step-out"}`}>
            <Narrator
              node={current.node}
              isRunning={!settled}
              vlm={vlm}
              decision={settled ? group.decision : undefined}
              iteration={group.iteration}
            />
            <NodeCard entry={current} decision={decisionFor(current)} zneFallback={zneFallback} folded={folded} lead />
          </div>
        )}
        {passed.map((n) =>
          expanded.has(n.node) ? (
            <NodeCard key={n.node} entry={n} decision={decisionFor(n)} zneFallback={zneFallback} folded={folded} onCollapse={() => toggle(n.node)} />
          ) : (
            <FoldedStep key={n.node} entry={n} onClick={() => toggle(n.node)} />
          )
        )}
      </div>
    </section>
  );
}

// Dr. Qubit — the first-person narrator above the current step. Picks a line by
// iteration index (deterministic), animates the ellipsis via CSS while running,
// and reacts to the decision once the iteration settles.
function Narrator({
  node,
  isRunning,
  vlm,
  decision,
  iteration,
}: {
  node: string;
  isRunning: boolean;
  vlm: boolean;
  decision?: DecisionEntry;
  iteration: number;
}) {
  if (!isRunning && decision) {
    const met = decision.action === "stop" && decision.metric_value != null && decision.target != null
      && decision.metric_value <= decision.target;
    const line = met
      ? "**Eureka!** Error's under the target — we did it! 🎉"
      : decision.action === "stop"
        ? "**Well now** — that's as far as physics will let us go. Calling it."
        : "**Hmm**, not there yet — but I have an idea. Let me push harder next round";
    return (
      <div className={`narrator settled ${met ? "met" : ""}`}>
        <span className="nr-avatar">🧑‍🔬</span>
        <div className="nr-body">
          <span className="nr-name">Dr. Qubit</span>
          <span className="nr-line">{renderRich(line)}</span>
        </div>
      </div>
    );
  }
  const visionLine = vlm ? NARRATION_VLM[node] : undefined;
  const pool = NARRATION[node] || ["Let me work through this step"];
  const line = visionLine ?? pool[iteration % pool.length];
  return (
    <div className="narrator">
      <span className="nr-avatar">🧑‍🔬</span>
      <div className="nr-body">
        <span className="nr-name">Dr. Qubit</span>
        <span className="nr-line">
          {renderRich(line)}
          <span className="dots" />
        </span>
      </div>
    </div>
  );
}

// Render a string with **bold** segments as <strong>, nothing else (no markdown lib).
function renderRich(text: string): ReactNode {
  return text.split(/(\*\*[^*]+\*\*)/g).map((seg, i) =>
    seg.startsWith("**") && seg.endsWith("**") ? <strong key={i}>{seg.slice(2, -2)}</strong> : <span key={i}>{seg}</span>
  );
}

// Live LLM transcript: the model's answer as it streams, token by token. While
// the step runs a caret blinks at the tail; once done it's the full answer.
function LiveStream({
  text,
  label,
  streaming,
}: {
  text: string;
  label: string;
  streaming: boolean;
}) {
  if (!text) return null;
  return (
    <div className={`llm-stream ${streaming ? "live" : ""}`}>
      <span className="ls-tag">{label}</span>
      <pre className="ls-body">
        {text}
        {streaming && <span className="ls-caret">▋</span>}
      </pre>
    </div>
  );
}

// The ZNE noise-folded circuit set — the per-iteration difference. The base
// logical circuit is identical every iteration; ZNE re-runs it gate-folded at
// scale 1, 3, 5, … (each deeper). Showing the set side-by-side makes the
// escalation visible (a single base diagram looked identical run to run).
function CircuitSet({
  folded,
  usesZne,
  running = false,
}: {
  folded: FoldedCircuit[];
  usesZne: boolean;
  running?: boolean;
}) {
  return (
    <>
      <p className="see-note">
        {running ? "Circuits running on the device" : "Circuits to run on the device"}
        {usesZne
          ? ` — ${folded.length} ZNE noise-scaled copies (deeper = more noise, extrapolated to zero):`
          : " — the target circuit (no ZNE folding this round):"}
      </p>
      <div className="circuit-set">
        {folded.map((f) => (
          <figure key={f.scale} className="circuit-fold">
            <figcaption>
              <span className="cf-scale">scale ×{f.scale}</span>
              <span className="cf-meta">depth {f.depth} · {f.n_gates} gates</span>
            </figcaption>
            <pre className="circuit-diagram">{f.diagram}</pre>
          </figure>
        ))}
      </div>
    </>
  );
}

// A passed step, collapsed: caret + badge + icon + label + one-line summary.
function FoldedStep({ entry, onClick }: { entry: NodeEntry; onClick: () => void }) {
  const meta = NODE_META[entry.node] || { n: "•", icon: "▫", label: entry.node };
  return (
    <button className={`folded-step ${entry.status}`} onClick={onClick}>
      <span className="fs-caret">▸</span>
      <span className="fs-badge">{meta.n}</span>
      <span className="fs-icon">{meta.icon}</span>
      <span className="fs-label">{meta.label}</span>
      <span className="fs-sum">{stepSummary(entry)}</span>
    </button>
  );
}

// A short one-liner per node for the folded row, from its streamed detail.
function stepSummary(entry: NodeEntry): string {
  const d = (entry.detail || {}) as Record<string, any>;
  switch (entry.node) {
    case "empirical_probe":
      return d.classification?.dominant_error ? `dominant → ${d.classification.dominant_error}` : "";
    case "strategy_select":
      return (d.strategy?.techniques || []).join(" + ");
    case "readout_calibrate":
      return d.skipped || !d.calibration ? "skipped" : `quality ${num(d.calibration.quality, 3)}`;
    case "circuit_generate":
      return d.plan?.n_executables ? `${d.plan.n_executables} executables` : "";
    case "execute":
      return d.value != null ? `⟨O⟩ ${num(d.value)} ± ${num(d.error_bar)}` : "";
    case "post_process":
      return d.estimate?.value != null ? `⟨O⟩ ${num(d.estimate.value)} ± ${num(d.estimate.error_bar)}` : "";
    case "validate":
      return d.metric_value != null ? `error ${num(d.metric_value)}` : "";
    default:
      return "";
  }
}

// Live error→target convergence, rendered just under the progress bar. A light
// inline SVG sparkline (no Plotly) so it's always-on and cheap.
function Convergence({ timeline, target }: { timeline: IterGroup[]; target: number }) {
  const points = timeline
    .map((g) => ({ iter: g.iteration, err: g.decision?.metric_value }))
    .filter((p): p is { iter: number; err: number } => typeof p.err === "number");
  if (points.length === 0) return null;

  const W = 360, H = 64, padX = 8, padY = 10;
  const maxErr = Math.max(...points.map((p) => p.err), target);
  const yMax = maxErr * 1.15 || 1;
  const n = points.length;
  const x = (i: number) => (n === 1 ? W / 2 : padX + (i * (W - 2 * padX)) / (n - 1));
  const y = (v: number) => padY + (1 - v / yMax) * (H - 2 * padY);
  const targetY = y(target);

  const last = points[points.length - 1].err;
  const prev = points.length >= 2 ? points[points.length - 2].err : null;
  const delta = last - target;
  const met = last <= target;
  const trend = prev == null ? "→" : last < prev - 1e-9 ? "↓" : last > prev + 1e-9 ? "↑" : "→";
  const trendClass = trend === "↓" ? "good" : trend === "↑" ? "bad" : "";

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.err).toFixed(1)}`).join(" ");

  return (
    <div className="convergence">
      <svg className="conv-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {/* target line */}
        <line x1={0} y1={targetY} x2={W} y2={targetY} className="conv-target" strokeDasharray="4 4" />
        <text x={W - 2} y={targetY - 3} className="conv-target-label" textAnchor="end">
          target {target.toFixed(4)}
        </text>
        {/* error trace */}
        {n >= 2 && <path d={path} className="conv-line" fill="none" />}
        {points.map((p, i) => (
          <circle key={p.iter} cx={x(i)} cy={y(p.err)} r={3.5} className={`conv-dot ${p.err <= target ? "met" : ""}`} />
        ))}
      </svg>
      <div className="conv-readout">
        <div className="cr-main mono">
          error {last.toFixed(4)} <span className="cr-arrow">→</span> target {target.toFixed(4)}
        </div>
        <div className="cr-sub">
          <span className={`conv-delta ${met ? "met" : "miss"}`}>
            {met ? "✓ met" : `Δ +${delta.toFixed(4)}`}
          </span>
          <span className={`conv-trend ${trendClass}`}>{trend}</span>
        </div>
      </div>
    </div>
  );
}

function NodeCard({
  entry,
  decision,
  zneFallback,
  folded,
  lead = false,
  onCollapse,
}: {
  entry: NodeEntry;
  decision?: DecisionEntry;
  zneFallback?: Figure; // ZNE figure from post_process, shown while validate runs
  folded?: FoldedCircuit[]; // ZNE folded circuit set, shown in the execute card
  lead?: boolean; // the current/active step, rendered at the top of the iteration
  onCollapse?: () => void; // present when this is a re-expanded passed step
}) {
  const meta = NODE_META[entry.node] || { n: "•", icon: "▫", label: entry.node };
  return (
    <div className={`node-card ${entry.status} ${lead ? "step-active" : ""}`}>
      <div className="nc-head">
        <span className="nc-badge">{meta.n}</span>
        <span className="nc-icon">{meta.icon}</span>
        <span className="nc-title">{meta.label}</span>
        <span className="dot" />
        {entry.shots ? <span className="nc-shots">{entry.shots.toLocaleString()} sh</span> : null}
        {onCollapse && (
          <button className="nc-collapse" onClick={onCollapse} title="collapse">▾</button>
        )}
      </div>
      <NodeBody entry={entry} decision={decision} zneFallback={zneFallback} folded={folded} />
    </div>
  );
}

// Renders the SEES / THINKS / DECIDES blocks per node from its detail + plots.
function NodeBody({
  entry,
  decision,
  zneFallback,
  folded,
}: {
  entry: NodeEntry;
  decision?: DecisionEntry;
  zneFallback?: Figure;
  folded?: FoldedCircuit[];
}) {
  const d = (entry.detail || {}) as Record<string, any>;
  const llm = entry.llm;

  if (entry.node === "empirical_probe") {
    const cls = d.classification || {};
    const vlm = d.vlm_classification as VlmTrace | null;
    const usedVlm = vlm && !vlm.degraded;
    const images = vlm?.images || [];
    const labels = ["readout probe (prep |0…0⟩)", "GHZ probe"];
    return (
      <>
        <Sees>
          {images.length > 0 ? (
            <>
              <p className="see-note">Exact image sent to Claude:</p>
              <div className="node-figs">
                {images.map((img, i) => (
                  <Chart key={i} image={img} title={labels[i]} bare />
                ))}
              </div>
            </>
          ) : (
            <div className="node-figs">
              {(entry.plots || []).map((pl) => (
                <Chart key={pl.name} figure={pl.data} bare height={240} />
              ))}
            </div>
          )}
        </Sees>
        <Thinks>
          <p>
            Readout mass off |0…0⟩ = <b>{num(cls.readout_error, 3)}</b> (flag &gt; 0.02);
            GHZ leakage = <b>{num(cls.ghz_leakage, 3)}</b> (gate error &gt; 0.05).
          </p>
          {/* Live: Claude's analysis as it streams, before the parsed verdict lands. */}
          {llm?.vlm && !usedVlm && (
            <LiveStream text={llm.vlm} label="Claude (vision)" streaming={entry.status === "running"} />
          )}
          {usedVlm && (
            <>
              {vlm!.prompt && <Prompt text={vlm!.prompt} />}
              <div className="claude-says">
                <span className="cs-tag">Claude</span>
                {vlm!.evidence || "(no evidence returned)"}
              </div>
            </>
          )}
        </Thinks>
        <Decides>
          dominant error → <b>{cls.dominant_error}</b> (confidence {cls.confidence}
          {cls.source === "vlm+rules" ? ", VLM" : ""}); prioritize{" "}
          <b>{(cls.suggested_focus || []).join(" + ") || "REM"}</b>
        </Decides>
      </>
    );
  }

  if (entry.node === "strategy_select") {
    const s = d.strategy || {};
    return (
      <>
        <Thinks>
          {d.source === "escalation"
            ? "Previous strategy missed the target — escalate to a stronger recipe."
            : "Pick the minimal sufficient mitigation for the dominant error."}
        </Thinks>
        <Decides>
          {d.source} → <b>{(s.techniques || []).join(" + ")}</b>
          {s.rem_twirls ? ` · ${s.rem_twirls} REM twirls` : ""}
        </Decides>
      </>
    );
  }

  if (entry.node === "readout_calibrate") {
    const c = d.calibration || {};
    if (d.skipped || !d.calibration) return <Decides>skipped (REM not in strategy)</Decides>;
    const errs = c.qubit_readout_errors as number[] | undefined;
    return (
      <Decides>
        built inverse confusion matrix · quality <b>{num(c.quality, 3)}</b>
        {errs ? ` · per-qubit readout err [${errs.map((e) => e.toFixed(3)).join(", ")}]` : ""}
        {c.rem_twirls ? ` · ${c.rem_twirls} twirls` : ""}
      </Decides>
    );
  }

  if (entry.node === "circuit_generate") {
    const p = d.plan || {};
    const tech = [p.uses_rem && "REM", p.uses_zne && "ZNE", p.uses_pt && "PT"].filter(Boolean).join(" + ");
    const fc = (d.folded_circuits as FoldedCircuit[] | undefined) || folded;
    return (
      <>
        {fc && fc.length > 0 && (
          <Sees>
            <CircuitSet folded={fc} usesZne={!!p.uses_zne} />
          </Sees>
        )}
        <Decides>
          compiled <b>{p.n_executables}</b> executables ({p.n_scales} ZNE scale(s) × {p.n_twirls} twirl(s) ×{" "}
          {p.n_bases} basis) · {tech}
        </Decides>
      </>
    );
  }

  if (entry.node === "execute") {
    return (
      <>
        {folded && folded.length > 0 && (
          <Sees>
            <CircuitSet folded={folded} usesZne={folded.length > 1} running={entry.status === "running"} />
          </Sees>
        )}
        <Decides>
          ⟨O⟩ = <b>{num(d.value)}</b> ± {num(d.error_bar)} · {(d.techniques || []).join(" + ")} ·{" "}
          {(d.shots_used || 0).toLocaleString()} shots
        </Decides>
      </>
    );
  }

  if (entry.node === "post_process") {
    const est = d.estimate || {};
    const hasZne = (entry.plots || []).length > 0;
    return (
      <>
        {hasZne && (
          <Sees>
            <div className="node-figs">
              {(entry.plots || []).map((pl) => (
                <Chart key={pl.name} figure={pl.data} bare height={260} />
              ))}
            </div>
          </Sees>
        )}
        <Decides>
          estimate ⟨O⟩ = <b>{num(est.value)}</b> ± {num(est.error_bar)}
        </Decides>
      </>
    );
  }

  if (entry.node === "validate") {
    const vlm = d.vlm_verdict as VlmTrace | null;
    const usedVlm = vlm && !vlm.degraded;
    const images = vlm?.images || [];
    const met = decision?.metric_value != null && decision.target != null && decision.metric_value <= decision.target;
    // The extrapolation the step inspects. Prefer the exact frame Claude saw;
    // else the validate node's own plot; else the post_process figure (available
    // *while* validate is still running, so the view is never blank).
    const ownFig = (entry.plots || []).find((p) => p.name === "zne_extrapolation")?.data;
    const figFallback = ownFig || zneFallback;
    return (
      <>
        {(images.length > 0 || figFallback) && (
          <Sees>
            <p className="see-note">
              {images.length > 0 ? "ZNE extrapolation Claude inspected:" : "ZNE extrapolation being inspected:"}
            </p>
            <div className="node-figs">
              {images.length > 0 ? (
                <Chart image={images[0]} title="zero-noise extrapolation" bare />
              ) : (
                <Chart figure={figFallback} bare height={260} />
              )}
            </div>
          </Sees>
        )}
        <Thinks>
          {decision?.metric_value != null && decision.target != null && (
            <p className="mono">
              error {decision.metric_value.toFixed(4)} {met ? "≤" : ">"} target {decision.target}
            </p>
          )}
          {/* Live: the VLM's plot analysis streaming in, before the parsed verdict. */}
          {llm?.vlm && !usedVlm && (
            <LiveStream text={llm.vlm} label="Claude (vision)" streaming={entry.status === "running"} />
          )}
          {usedVlm && (
            <>
              {vlm!.prompt && <Prompt text={vlm!.prompt} />}
              <div className="claude-says">
                <span className="cs-tag">Claude</span>
                {vlm!.rationale || "(no rationale returned)"}
                {typeof vlm!.confidence === "number" ? ` (confidence ${vlm!.confidence})` : ""}
              </div>
            </>
          )}
          {/* Live: the orchestration agent's reasoning as it decides stop/retry. */}
          {llm?.agent && (
            <LiveStream text={llm.agent} label="Orchestrator agent" streaming={entry.status === "running" && !decision} />
          )}
        </Thinks>
        {decision && (
          <div className={`decide ${met ? "stop" : "retry"}`}>
            <span className="block-tag">🎯 DECIDES</span>
            <b>{decision.action}</b>
            {decision.source && decision.source !== "rules" ? ` (${decision.source})` : ""}: {decision.reason}
          </div>
        )}
      </>
    );
  }

  if (entry.node === "report") {
    return <Decides>assembled final estimate, audit trail, and efficiency comparison</Decides>;
  }

  return null;
}

// SEES / THINKS / DECIDES block wrappers.
function Sees({ children }: { children: ReactNode }) {
  return <div className="block see"><span className="block-tag">👁 SEES</span><div className="block-body">{children}</div></div>;
}
function Thinks({ children }: { children: ReactNode }) {
  return <div className="block think"><span className="block-tag">🧠 THINKS</span><div className="block-body">{children}</div></div>;
}
function Decides({ children }: { children: ReactNode }) {
  return <div className="block decide"><span className="block-tag">🎯 DECIDES</span><div className="block-body">{children}</div></div>;
}

function Prompt({ text }: { text: string }) {
  return (
    <details className="vlm-prompt">
      <summary>prompt sent to Claude</summary>
      <pre>{text}</pre>
    </details>
  );
}

function Results({ result }: { result: RunResult }) {
  const est = result.estimate;
  const cmp = result.comparison;
  const err = est ? Math.abs(est.value - result.ideal) : null;
  const met = err !== null && err <= result.target_accuracy;

  return (
    <>
      <div className="panel result-head">
        <h2>Result — {result.status} · {result.iterations} iteration(s)</h2>
        <div className="cards">
          <div className="card"><div className="k">Estimate</div>
            <div className="v">{est ? est.value.toFixed(4) : "—"}</div></div>
          <div className="card"><div className="k">Error vs ideal</div>
            <div className={`v ${met ? "good" : "bad"}`}>{err !== null ? err.toFixed(4) : "—"}</div></div>
          <div className="card"><div className="k">Shots used</div>
            <div className="v">{result.shots_used?.toLocaleString()}</div></div>
          <div className="card"><div className="k">Techniques</div>
            <div className="v" style={{ fontSize: 15 }}>{est?.techniques.join(" + ") || "—"}</div></div>
        </div>
      </div>

      {cmp && (
        <div className="panel">
          <h2>Efficiency vs blind full-stack baseline</h2>
          <div className="cards">
            <div className="card"><div className="k">Adaptive shots</div>
              <div className="v good">{cmp.adaptive.shots.toLocaleString()}</div></div>
            <div className="card"><div className="k">Baseline shots</div>
              <div className="v">{cmp.baseline.shots.toLocaleString()}</div></div>
            <div className="card"><div className="k">Shot ratio</div>
              <div className="v good">{cmp.shot_ratio ? cmp.shot_ratio.toFixed(2) + "×" : "—"}</div></div>
            <div className="card"><div className="k">Efficiency gain</div>
              <div className="v"><span className={`badge ${cmp.efficiency_gain_demonstrated ? "good" : "bad"}`}>
                {cmp.efficiency_gain_demonstrated ? "DEMONSTRATED" : "not shown"}</span></div></div>
          </div>
        </div>
      )}

      <div className="charts">
        {result.figures.zne && <div className="chart-full"><Chart figure={result.figures.zne} /></div>}
        {result.figures.accuracy_vs_shots && (
          <div className="chart-full"><Chart figure={result.figures.accuracy_vs_shots} /></div>
        )}
      </div>

      <div className="panel">
        <h2>Policy audit ({result.audit.length})</h2>
        <table className="audit">
          <thead><tr><th>Node</th><th>Action</th><th>Pred. shots</th><th>Approved</th><th>Reason</th></tr></thead>
          <tbody>
            {result.audit.map((a, i) => (
              <tr key={i}>
                <td>{a.node_id}</td>
                <td>{a.action}</td>
                <td>{a.predicted_shots.toLocaleString()}</td>
                <td className={a.approved ? "ok" : "rej"}>{a.approved ? "✓" : "✗"}</td>
                <td style={{ color: "var(--muted)" }}>{a.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
