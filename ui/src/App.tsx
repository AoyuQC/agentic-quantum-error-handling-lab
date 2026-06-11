import { useEffect, useState } from "react";
import { fetchDevices, runStream } from "./api";
import { Chart } from "./Chart";
import { NODES, type ProgressEvent, type RunRequest, type RunResult } from "./types";

type NodeState = "idle" | "running" | "done" | "failed";

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
  const [nodeState, setNodeState] = useState<Record<string, NodeState>>({});
  const [nodeShots, setNodeShots] = useState<Record<string, number>>({});
  const [log, setLog] = useState<string[]>([]);
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

  function onProgress(ev: ProgressEvent) {
    if (ev.event === "node_start" && ev.node) {
      setNodeState((s) => ({ ...s, [ev.node!]: "running" }));
    } else if (ev.event === "node_done" && ev.node) {
      setNodeState((s) => ({ ...s, [ev.node!]: "done" }));
      if (ev.shots_used) setNodeShots((s) => ({ ...s, [ev.node!]: (s[ev.node!] || 0) + ev.shots_used! }));
    } else if (ev.event === "node_failed" && ev.node) {
      setNodeState((s) => ({ ...s, [ev.node!]: "failed" }));
    } else if (ev.event === "decision") {
      setLog((l) => [...l, `iter ${ev.iteration}: ${ev.action} — ${ev.reason}`]);
    }
  }

  async function start() {
    setRunning(true);
    setResult(null);
    setError(null);
    setLog([]);
    setNodeShots({});
    setNodeState(Object.fromEntries(NODES.map((n) => [n, "idle"])));
    try {
      await runStream(req, {
        onProgress,
        onResult: setResult,
        onError: setError,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Agentic Quantum Error Mitigation</h1>
        <p>VLM-guided adaptive QEM on Amazon Braket — probe, mitigate, early-stop, prove the efficiency gain.</p>
      </header>

      <div className="layout">
        {/* Controls + live progress */}
        <div>
          <div className="panel">
            <h2>Run</h2>
            <label>Qubits</label>
            <input type="number" min={1} max={6} value={req.qubits}
              onChange={(e) => set("qubits", +e.target.value)} />

            <label>Target accuracy</label>
            <input type="number" step={0.01} value={req.target_accuracy}
              onChange={(e) => set("target_accuracy", +e.target.value)} />

            <label>Device (noise model)</label>
            <select value={req.device} onChange={(e) => set("device", e.target.value)}>
              {devices.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>

            <label>Seed</label>
            <input type="number" value={req.seed ?? 7} onChange={(e) => set("seed", +e.target.value)} />

            <div className="checkbox">
              <input type="checkbox" id="vlm" checked={req.use_vlm}
                onChange={(e) => set("use_vlm", e.target.checked)} />
              <label htmlFor="vlm" style={{ margin: 0 }}>Use Bedrock Claude VLM</label>
            </div>
            <div className="checkbox">
              <input type="checkbox" id="cmp" checked={req.compare_baseline}
                onChange={(e) => set("compare_baseline", e.target.checked)} />
              <label htmlFor="cmp" style={{ margin: 0 }}>Compare vs full-stack baseline</label>
            </div>

            <button className="run" onClick={start} disabled={running}>
              {running ? "Running…" : "Run adaptive loop"}
            </button>
          </div>

          <div className="panel">
            <h2>DAG progress</h2>
            <div className="nodes">
              {NODES.map((n) => (
                <div key={n} className={`node-row ${nodeState[n] || "idle"}`}>
                  <span className="dot" />
                  <span>{n}</span>
                  {nodeShots[n] ? <span className="shots">{nodeShots[n].toLocaleString()} sh</span> : null}
                </div>
              ))}
            </div>
            {log.length > 0 && (
              <div className="log" style={{ marginTop: 12 }}>
                {log.map((l, i) => <div key={i}>{l}</div>)}
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <div>
          {error && <div className="panel" style={{ color: "var(--danger)" }}>Error: {error}</div>}

          {!result && !error && (
            <div className="panel empty">
              Configure a run and press <strong>Run adaptive loop</strong>. Watch the DAG progress
              on the left; charts and the efficiency comparison appear here.
            </div>
          )}

          {result && <Results result={result} />}
        </div>
      </div>
    </div>
  );
}

function Results({ result }: { result: RunResult }) {
  const est = result.estimate;
  const cmp = result.comparison;
  const err = est ? Math.abs(est.value - result.ideal) : null;
  const met = err !== null && err <= result.target_accuracy;

  return (
    <>
      <div className="panel">
        <h2>Result — {result.status}</h2>
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
        {result.figures.classification && (
          <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 12 }}>
            Probe classification: <strong>{result.figures.classification.dominant_error}</strong>
            {" "}(confidence {result.figures.classification.confidence},
            {" "}{result.figures.classification.source || "rules"})
            {result.vlm_used ? " · VLM on" : " · rules-only"}
          </p>
        )}
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
        <Chart figure={result.figures.readout_probe} />
        <Chart figure={result.figures.ghz_probe} />
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
