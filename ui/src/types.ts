export interface RunRequest {
  qubits: number;
  target_accuracy: number;
  device: string;
  budget_shots: number;
  use_vlm: boolean;
  compare_baseline: boolean;
  seed: number | null;
}

// Plotly figure: { data, layout } as plain JSON.
export type Figure = { data: unknown[]; layout: Record<string, unknown> };

// A figure a node produced (what the agent "sees").
export interface PlotRecord {
  name: string;
  format: string; // "plotly"
  data: Figure;
}

// The transparent trace of a VLM call: the exact image(s) and prompt sent to
// Claude, its raw answer, and the parsed structured reasoning. Shapes differ
// slightly between the probe (evidence/dominant_error) and validate
// (rationale/recommended_action) calls, so fields are all optional.
export interface VlmTrace {
  prompt?: string;
  images?: string[]; // base64 PNGs
  raw_response?: string;
  degraded?: boolean;
  reason?: string;
  confidence?: number;
  // probe classification
  dominant_error?: string;
  readout_asymmetry?: boolean;
  evidence?: string;
  suggested_focus?: string[];
  // validate decision
  extrapolation_monotone?: boolean;
  has_outliers?: boolean;
  readout_anomaly?: boolean;
  improvement_meaningful?: boolean;
  recommended_action?: string;
  rationale?: string;
}

export interface ProgressEvent {
  event: string;
  node?: string;
  iteration?: number;
  status?: string;
  shots_used?: number;
  action?: string;
  reason?: string;
  nodes?: string[];
  // llm_delta: a streamed text fragment from the VLM ("vlm") or the
  // orchestration agent ("agent"), surfaced live while the node runs.
  role?: string;
  delta?: string;
  // node_done: the node's outputs minus large count arrays (agent reasoning).
  detail?: Record<string, unknown>;
  // node_done: figures the node produced (what the agent sees).
  plots?: PlotRecord[];
  // decision: provenance + the metric compared to target.
  source?: string;
  metric_value?: number | null;
  error_estimate?: number | null;
  target?: number;
  // cache_status: whether this run is a replay of a recorded run (true) or a
  // fresh, live run being recorded (false).
  cached?: boolean;
}

export interface Experiment {
  num_qubits: number;
  description: string;
  observable_terms: [number, string][];
  ansatz: string;
  ideal: number;
  target_accuracy: number;
  device: string;
  budget_shots: number;
  seed: number | null;
}

export interface AuditRecord {
  node_id: string;
  action: string;
  approved: boolean;
  predicted_shots: number;
  reason: string;
  budget_remaining_shots: number | null;
}

export interface Estimate {
  value: number;
  error_bar: number;
  shots_used: number;
  techniques: string[];
  zne_data: Record<string, number>;
}

export interface Comparison {
  ideal: number;
  target_accuracy: number;
  adaptive: { shots: number; error: number; techniques: string[] };
  baseline: { shots: number; error: number; techniques: string[] };
  shot_ratio: number | null;
  shots_saved: number;
  efficiency_gain_demonstrated: boolean;
  adaptive_meets_target: boolean;
}

export interface RunResult {
  status: string;
  iterations: number;
  experiment?: Experiment;
  device: string;
  ideal: number;
  target_accuracy: number;
  estimate: Estimate | null;
  shots_used: number;
  decision: { action: string; reason: string } | null;
  audit: AuditRecord[];
  figures: {
    readout_probe?: Figure;
    ghz_probe?: Figure;
    zne?: Figure;
    accuracy_vs_shots?: Figure;
    classification?: { dominant_error: string; confidence: number; source?: string };
  };
  comparison?: Comparison;
  vlm_used: boolean;
}

// The eight DAG stages, in pipeline order.
export const NODES = [
  "empirical_probe",
  "strategy_select",
  "readout_calibrate",
  "circuit_generate",
  "execute",
  "post_process",
  "validate",
  "report",
] as const;
