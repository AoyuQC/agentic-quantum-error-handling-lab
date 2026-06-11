export interface RunRequest {
  qubits: number;
  target_accuracy: number;
  device: string;
  budget_shots: number;
  use_vlm: boolean;
  compare_baseline: boolean;
  seed: number | null;
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

// Plotly figure: { data, layout } as plain JSON.
export type Figure = { data: unknown[]; layout: Record<string, unknown> };

export interface RunResult {
  status: string;
  iterations: number;
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
