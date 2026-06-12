import { useEffect, useState, type ReactNode } from "react";
import { fetchDevices, runStream } from "./api";
import { Chart } from "./Chart";
import type {
  Experiment,
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
};
type DecisionEntry = {
  action: string;
  reason: string;
  source?: string;
  metric_value?: number | null;
  target?: number;
};
type IterGroup = { iteration: number; nodes: NodeEntry[]; decision?: DecisionEntry };

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
  // Which iteration is expanded. null => follow the newest one automatically
  // (older iterations auto-fold as the loop advances). A click pins one open.
  const [expandedIter, setExpandedIter] = useState<number | null>(null);
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
        // A new iteration started — fold the previous ones and follow this one.
        setExpandedIter(null);
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
    try {
      await runStream(req, { onProgress, onResult: setResult, onError: setError });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  // Which iteration occupies the center stage: the user-selected one, else the
  // newest (older ones auto-fold into the side rail as the loop advances).
  const shownIter =
    expandedIter != null && timeline.some((g) => g.iteration === expandedIter)
      ? expandedIter
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

        {shown && <IterationStage group={shown} />}

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

function IterationStage({ group }: { group: IterGroup }) {
  const met = group.decision?.action === "stop";
  return (
    <section className="iter-section open">
      <h2 className={`iter-title static ${met ? "met" : ""}`}>
        <span>▸ Iteration {group.iteration}</span>
        {group.decision && <span className="iter-summary">{iterSummary(group)}</span>}
      </h2>
      <div className="node-cards">
        {group.nodes.map((n) => (
          <NodeCard key={n.node} entry={n} decision={n.node === "validate" ? group.decision : undefined} />
        ))}
      </div>
    </section>
  );
}

function NodeCard({ entry, decision }: { entry: NodeEntry; decision?: DecisionEntry }) {
  const meta = NODE_META[entry.node] || { n: "•", icon: "▫", label: entry.node };
  return (
    <div className={`node-card ${entry.status}`}>
      <div className="nc-head">
        <span className="nc-badge">{meta.n}</span>
        <span className="nc-icon">{meta.icon}</span>
        <span className="nc-title">{meta.label}</span>
        <span className="dot" />
        {entry.shots ? <span className="nc-shots">{entry.shots.toLocaleString()} sh</span> : null}
      </div>
      <NodeBody entry={entry} decision={decision} />
    </div>
  );
}

// Renders the SEES / THINKS / DECIDES blocks per node from its detail + plots.
function NodeBody({ entry, decision }: { entry: NodeEntry; decision?: DecisionEntry }) {
  const d = (entry.detail || {}) as Record<string, any>;

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
    return (
      <Decides>
        compiled <b>{p.n_executables}</b> executables ({p.n_scales} ZNE scale(s) × {p.n_twirls} twirl(s) ×{" "}
        {p.n_bases} basis) · {tech}
      </Decides>
    );
  }

  if (entry.node === "execute") {
    return (
      <Decides>
        ⟨O⟩ = <b>{num(d.value)}</b> ± {num(d.error_bar)} · {(d.techniques || []).join(" + ")} ·{" "}
        {(d.shots_used || 0).toLocaleString()} shots
      </Decides>
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
    return (
      <>
        {images.length > 0 && (
          <Sees>
            <p className="see-note">ZNE extrapolation Claude inspected:</p>
            <div className="node-figs">
              <Chart image={images[0]} title="zero-noise extrapolation" bare />
            </div>
          </Sees>
        )}
        <Thinks>
          {decision?.metric_value != null && decision.target != null && (
            <p className="mono">
              error {decision.metric_value.toFixed(4)} {met ? "≤" : ">"} target {decision.target}
            </p>
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
