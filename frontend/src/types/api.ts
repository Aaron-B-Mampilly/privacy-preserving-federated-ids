/** Mirrors api/schemas.py. Kept hand-written and small rather than
 *  generated, so the shapes the UI actually consumes stay obvious. */

export type Provenance = 'existing' | 'new' | 'derived' | 'live' | 'demo';

/** `mean: null` means NOT MEASURED for this scope. It never means zero. */
export interface MetricValue {
  mean: number | null;
  std: number | null;
  n: number;
  provenance: Provenance;
  note?: string | null;
}

export interface ScopeInfo {
  dataset: string;
  scope: string;
  label: string;
  num_clients: number;
  clients_per_round: number;
  num_features: number;
}

export interface DatasetInfo {
  id: string;
  label: string;
  description: string;
  scopes: string[];
  scope_label: string;
  rare_labels: string[];
  zero_day_holdout: string[];
  available: boolean;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  results_available: number;
  checkpoints_available: number;
  consolidated_tables: boolean;
  datasets: string[];
  grafana_url: string;
  prometheus_url: string;
  notes: string[];
}

export interface SeriesPoint {
  round: number;
  value: number | null;
}

export interface ChartSeries {
  key: string;
  label: string;
  points: SeriesPoint[];
  provenance: Provenance;
  unit?: string | null;
}

export interface EventItem {
  kind: 'info' | 'drift' | 'retrain' | 'privacy' | 'round' | 'warning';
  title: string;
  detail?: string | null;
  round?: number | null;
  provenance: Provenance;
}

export interface DashboardResponse {
  scope: ScopeInfo;
  source_run: string;
  kpis: Record<string, MetricValue>;
  series: ChartSeries[];
  events: EventItem[];
  model_status: Record<string, unknown>;
}

export interface ExperimentSummary {
  id: string;
  title: string;
  subtitle: string;
  table_ref: string;
  available: boolean;
}

export interface ExperimentDetail {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  interpretation: string[];
  data: Record<string, any>;
  scopes_covered: string[];
  provenance_note: string;
}

export interface FLRunSummary {
  run_name: string;
  mechanism: string;
  label: string;
  rounds: number;
  best_round: number | null;
  clients_configured: number;
  clients_per_round: number;
  local_epochs: number | null;
  accuracy: MetricValue;
  macro_f1: MetricValue;
  mb_per_round: number | null;
}

export interface FLRunDetail {
  run_name: string;
  mechanism: string;
  scope: ScopeInfo;
  config: Record<string, any>;
  series: ChartSeries[];
  clients: Array<{
    client_id: number;
    status: string;
    participated: boolean;
    num_samples: number | null;
    accuracy: number | null;
    macro_f1: number | null;
  }>;
  best_round: number | null;
  convergence_note: string;
}

export interface PrivacyResponse {
  scope: ScopeInfo;
  differential_privacy: Record<string, any>;
  secure_aggregation: Record<string, any>;
  utility_curve: Array<Record<string, any>>;
  attacks: Record<string, any>;
}

export type DriftState =
  | 'normal' | 'warning' | 'drift_detected' | 'retraining' | 'recovered' | 'unknown';

export interface DriftStatusResponse {
  scope: ScopeInfo;
  state: DriftState;
  detector: Record<string, any>;
  recovery: Record<string, any>;
  rounds: Array<{
    round: number;
    psi: number | null;
    psi_flag: boolean;
    adwin_flag: boolean;
    mean_drift: number | null;
    retrain_triggered: boolean;
  }>;
  active_job: RetrainJob | null;
}

export interface RetrainJob {
  job_id: string;
  state: 'queued' | 'running' | 'completed' | 'failed';
  dataset: string;
  scope: string;
  run_name: string;
  sandboxed: boolean;
  started_at: string;
  finished_at: string | null;
  message: string | null;
  result: Record<string, any> | null;
}

export interface SampleItem {
  sequence_index: number;
  label: string;
  client_id: number;
  split: string;
}

export interface SamplesResponse {
  scope: ScopeInfo;
  total: number;
  samples: SampleItem[];
  labels_available: string[];
}

export interface PredictionResponse {
  prediction_id: string;
  timestamp: string;
  scope: ScopeInfo;
  model_run: string;
  mechanism: string;
  client_id: number | null;
  source: 'prepared_sample' | 'uploaded_csv' | 'raw_window';
  sequence_index: number | null;
  true_label: string | null;
  predicted_label: string;
  confidence: number;
  threat_level: 'benign' | 'low' | 'medium' | 'high' | 'unknown';
  is_attack: boolean;
  is_new_class: boolean;
  new_class_distance: number | null;
  new_class_threshold: number | null;
  prototype_label: string | null;
  reconstruction_mse: number;
  top_k: Array<{ label: string; probability: number }>;
  provenance: Provenance;
}

/** Normalised error the UI renders. Never contains a stack trace. */
export interface ApiError {
  error: string;
  detail: string;
  hint?: string | null;
  status: number;
  offline: boolean;
}
