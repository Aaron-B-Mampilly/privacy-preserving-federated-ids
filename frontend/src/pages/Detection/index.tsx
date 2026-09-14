import { useState } from 'react';
import { ScanSearch } from 'lucide-react';
import { api, toApiError } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import {
  Badge,
  Button,
  Card,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorState,
  Tabs,
  fmtPercent,
} from '@/components/primitives';
import type { Column } from '@/components/primitives';
import type { ApiError, PredictionResponse, SampleItem } from '@/types/api';
import SamplePicker from './SamplePicker';
import CsvUpload from './CsvUpload';
import ResultCard from './ResultCard';

const MECHANISMS = [
  { id: 'personalized', label: 'Ours — Personalized FL' },
  { id: 'fedavg', label: 'FedAvg' },
];

type Source = 'sample' | 'upload';

export default function Detection() {
  const { dataset, scope } = useDataset();
  const [source, setSource] = useState<Source>('sample');
  const [sample, setSample] = useState<SampleItem | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [mechanism, setMechanism] = useState('personalized');

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<ApiError | null>(null);

  const { data: history, refresh: refreshHistory } = useApi<PredictionResponse[]>(
    () => api.predictions(15),
    [],
  );

  const canAnalyze = source === 'sample' ? sample !== null : file !== null;

  async function analyze() {
    setRunning(true);
    setRunError(null);
    try {
      const prediction =
        source === 'sample' && sample
          ? await api.predict({
              dataset, scope,
              sequence_index: sample.sequence_index,
              client_id: sample.client_id >= 0 ? sample.client_id : undefined,
              mechanism,
            })
          : await api.predictUpload(file as File, dataset, scope, mechanism);
      setResult(prediction);
      refreshHistory();
    } catch (err) {
      setRunError(toApiError(err));
    } finally {
      setRunning(false);
    }
  }

  const historyColumns: Column<PredictionResponse>[] = [
    { key: 'time', header: 'Time', render: (r) => <span className="font-mono text-xs">{new Date(r.timestamp).toLocaleTimeString()}</span> },
    { key: 'source', header: 'Source', render: (r) => <span className="text-xs capitalize">{r.source.replace('_', ' ')}</span> },
    { key: 'true', header: 'True label', render: (r) => <span className="text-xs">{r.true_label ?? '—'}</span> },
    {
      key: 'pred',
      header: 'Predicted',
      render: (r) => (
        <span className="flex items-center gap-1.5 text-xs">
          {r.predicted_label}
          {r.is_new_class && <Badge tone="warning">NEW CLASS</Badge>}
        </span>
      ),
    },
    { key: 'conf', header: 'Confidence', align: 'right', render: (r) => <span className="font-mono text-xs">{fmtPercent(r.confidence)}</span> },
    {
      key: 'threat',
      header: 'Threat',
      render: (r) => (
        <Badge tone={r.is_new_class ? 'warning' : r.is_attack ? 'danger' : 'good'}>
          {r.is_new_class ? 'new class' : r.is_attack ? 'attack' : 'benign'}
        </Badge>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <ScanSearch size={22} className="text-accent" />
          <h1 className="font-display text-2xl">Intrusion Detection</h1>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Real inference through a trained checkpoint — on a real held-out sample, a true zero-day
          holdout sample, or your own preprocessed CSV window.
        </p>
      </header>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader
              title="Input"
              action={
                <select
                  value={mechanism}
                  onChange={(e) => setMechanism(e.target.value)}
                  className="rounded-md border border-border bg-surface px-2 py-1 text-xs"
                >
                  {MECHANISMS.map((m) => (
                    <option key={m.id} value={m.id}>{m.label}</option>
                  ))}
                </select>
              }
            />
            <Tabs
              tabs={[
                { id: 'sample' as Source, label: 'Prepared sample' },
                { id: 'upload' as Source, label: 'Upload CSV' },
              ]}
              active={source}
              onChange={setSource}
            />
            <div className="mt-4">
              {source === 'sample' ? (
                <SamplePicker selected={sample} onSelect={setSample} />
              ) : (
                <CsvUpload file={file} onFile={setFile} />
              )}
            </div>
            <Button
              variant="primary"
              className="mt-4 w-full"
              disabled={!canAnalyze || running}
              onClick={analyze}
            >
              {running ? 'Analyzing…' : 'Analyze Traffic'}
            </Button>
            {sample && source === 'sample' && (
              <p className="mt-2 text-center text-2xs text-ink-muted">
                Selected: sequence #{sample.sequence_index} · client {sample.client_id} · label {sample.label}
              </p>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          {runError && <ErrorState error={runError} onRetry={analyze} />}
          {!runError && result && <ResultCard result={result} />}
          {!runError && !result && (
            <Card>
              <EmptyState
                title="No prediction yet"
                detail="Pick a prepared sample or upload a CSV, then run Analyze Traffic."
              />
            </Card>
          )}
        </div>
      </div>

      <Card>
        <CardHeader title="Prediction history" subtitle="This API process's session — not a persisted audit log" />
        <DataTable
          columns={historyColumns}
          rows={history ?? []}
          rowKey={(r) => r.prediction_id}
          emptyMessage="No predictions run yet."
        />
      </Card>
    </div>
  );
}
