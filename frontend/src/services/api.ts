import axios, { AxiosError } from 'axios';
import type {
  ApiError,
  DashboardResponse,
  DatasetInfo,
  DriftStatusResponse,
  ExperimentDetail,
  ExperimentSummary,
  FLRunDetail,
  FLRunSummary,
  HealthResponse,
  PredictionResponse,
  PrivacyResponse,
  RetrainJob,
  SamplesResponse,
} from '@/types/api';

const baseURL = import.meta.env.VITE_API_BASE ?? 'http://localhost:8001/api';

export const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL ?? 'http://localhost:3000';

const http = axios.create({ baseURL, timeout: 30_000 });

/**
 * Turns anything axios throws into a flat, user-safe shape. The backend
 * already refuses to send tracebacks; this is the second half of that
 * contract on the client side -- the UI only ever renders `detail`/`hint`.
 */
export function toApiError(error: unknown): ApiError {
  const axiosError = error as AxiosError<any>;

  if (axiosError?.code === 'ERR_NETWORK' || axiosError?.code === 'ECONNABORTED') {
    return {
      error: 'Backend unavailable',
      detail: 'Could not reach the FedPDA-IDS API.',
      hint: `Start the FastAPI server (uvicorn api.main:app --port 8001) and try again. Expected at ${baseURL}.`,
      status: 0,
      offline: true,
    };
  }

  const status = axiosError?.response?.status ?? 500;
  const payload = axiosError?.response?.data;

  if (payload && typeof payload === 'object') {
    return {
      error: payload.error ?? (status === 404 ? 'Not found' : 'Request failed'),
      detail: payload.detail ?? 'The API rejected this request.',
      hint: payload.hint ?? null,
      status,
      offline: false,
    };
  }

  return {
    error: 'Request failed',
    detail: `The API returned status ${status}.`,
    hint: null,
    status,
    offline: false,
  };
}

async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const response = await http.get<T>(url, { params });
  return response.data;
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const response = await http.post<T>(url, body);
  return response.data;
}

export const api = {
  health: () => get<HealthResponse>('/health'),
  datasets: () => get<DatasetInfo[]>('/datasets'),

  dashboard: (dataset: string, scope: string) =>
    get<DashboardResponse>('/dashboard', { dataset, scope }),

  experiments: () => get<ExperimentSummary[]>('/results'),
  experiment: (id: string) => get<ExperimentDetail>(`/results/${id}`),

  flRuns: (dataset: string, scope: string) =>
    get<FLRunSummary[]>('/fl/runs', { dataset, scope }),
  flRun: (runName: string) => get<FLRunDetail>(`/fl/runs/${runName}`),

  privacy: (dataset: string, scope: string) =>
    get<PrivacyResponse>('/privacy', { dataset, scope }),
  privacyUtility: (dataset: string, scope: string) =>
    get<{ scope: any; curve: any[]; note: string }>('/privacy/utility', { dataset, scope }),

  driftStatus: (dataset: string, scope: string) =>
    get<DriftStatusResponse>('/drift/status', { dataset, scope }),
  driftEvents: (dataset: string, scope: string) =>
    get<{ scope: any; events: any[] }>('/drift/events', { dataset, scope }),
  triggerRetrain: (dataset: string, scope: string, rounds: number) =>
    post<RetrainJob>('/drift/retrain', { dataset, scope, num_retrain_rounds: rounds }),
  retrainStatus: (jobId: string) => get<RetrainJob>(`/drift/retrain/${jobId}`),

  samples: (params: {
    dataset: string;
    scope: string;
    label?: string;
    client_id?: number;
    split?: string;
    limit?: number;
  }) => get<SamplesResponse>('/samples', params),

  predict: (body: {
    dataset: string;
    scope: string;
    sequence_index?: number;
    client_id?: number;
    mechanism?: string;
  }) => post<PredictionResponse>('/predict', body),

  predictUpload: async (file: File, dataset: string, scope: string, mechanism = 'personalized') => {
    const form = new FormData();
    form.append('file', file);
    form.append('dataset', dataset);
    form.append('scope', scope);
    form.append('mechanism', mechanism);
    const response = await http.post<PredictionResponse>('/predict/upload', form);
    return response.data;
  },

  predictions: (limit = 25) => get<PredictionResponse[]>('/predictions', { limit }),
};

export { baseURL as API_BASE_URL };
