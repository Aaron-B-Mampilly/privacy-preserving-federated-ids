import { useEffect, useState } from 'react';
import { Cpu, Users } from 'lucide-react';
import { api } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import {
  Badge,
  Card,
  CardHeader,
  Callout,
  DataTable,
  EmptyState,
  ErrorState,
  LoadingState,
  StatTile,
  cx,
  fmt,
} from '@/components/primitives';
import type { Column } from '@/components/primitives';
import { ChartFrame, SeriesLineChart } from '@/charts/ChartFrame';
import type { FLRunDetail, FLRunSummary } from '@/types/api';

const SERIES_LABELS: Record<string, string> = {
  macro_f1: 'Macro-F1',
  accuracy: 'Accuracy',
  val_loss: 'Centralized val loss/MSE',
  client_loss: 'Distributed client loss',
  mb_per_round: 'Comms MB/round',
};

export default function FederatedLearning() {
  const { dataset, scope } = useDataset();
  const [activeRun, setActiveRun] = useState<string | null>(null);

  const { data: runs, error: runsError, loading: runsLoading, refresh: refreshRuns } = useApi<FLRunSummary[]>(
    () => api.flRuns(dataset, scope),
    [dataset, scope],
  );

  useEffect(() => {
    // Default to this scope's personalized run (the headline mechanism)
    // whenever the dataset/scope changes and the current selection no
    // longer belongs to it.
    if (!runs?.length) return;
    if (!activeRun || !runs.some((r) => r.run_name === activeRun)) {
      const preferred = runs.find((r) => r.mechanism === 'personalized') ?? runs[0];
      setActiveRun(preferred.run_name);
    }
  }, [runs, activeRun]);

  const { data: detail, error: detailError, loading: detailLoading, refresh: refreshDetail } = useApi<FLRunDetail>(
    () => api.flRun(activeRun as string),
    [activeRun],
    { enabled: !!activeRun },
  );

  const clientColumns: Column<FLRunDetail['clients'][number]>[] = [
    { key: 'id', header: 'Client', render: (c) => <span className="font-mono">#{c.client_id}</span> },
    {
      key: 'status',
      header: 'Status',
      render: (c) => (
        <Badge tone={c.participated ? 'good' : 'neutral'}>{c.participated ? 'evaluated' : c.status.replace(/_/g, ' ')}</Badge>
      ),
    },
    { key: 'samples', header: 'Test samples', align: 'right', render: (c) => <span>{c.num_samples ?? '—'}</span> },
    { key: 'acc', header: 'Accuracy', align: 'right', render: (c) => <span>{c.accuracy != null ? fmt(c.accuracy, 4) : '—'}</span> },
    { key: 'f1', header: 'Macro-F1', align: 'right', render: (c) => <span>{c.macro_f1 != null ? fmt(c.macro_f1, 4) : '—'}</span> },
  ];

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Cpu size={22} className="text-accent" />
          <h1 className="font-display text-2xl">Federated Learning</h1>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Completed FL runs, replayed from their saved round-by-round history — training in this
          project runs offline in multi-hour batches, so this is not a live trainer.
        </p>
      </header>

      {runsLoading && !runs && <LoadingState label="Loading runs" />}
      {runsError && <ErrorState error={runsError} onRetry={refreshRuns} />}

      {runs && runs.length === 0 && (
        <EmptyState title="No completed runs for this scope" />
      )}

      {runs && runs.length > 0 && (
        <>
          <div className="flex flex-wrap gap-2">
            {runs.map((run) => (
              <button
                key={run.run_name}
                type="button"
                onClick={() => setActiveRun(run.run_name)}
                className={cx(
                  'rounded-md border px-3 py-1.5 text-left text-sm transition-colors',
                  activeRun === run.run_name
                    ? 'border-accent bg-accent-soft text-accent-strong'
                    : 'border-border bg-surface hover:bg-surface-alt',
                )}
              >
                <span className="font-medium">{run.label}</span>
                <span className="ml-2 font-mono text-xs text-ink-muted">f1={fmt(run.macro_f1.mean, 3)}</span>
              </button>
            ))}
          </div>

          {detailLoading && !detail && <LoadingState label="Loading run detail" />}
          {detailError && <ErrorState error={detailError} onRetry={refreshDetail} />}

          {detail && (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <StatTile label="Rounds" metric={{ mean: detail.config.num_rounds, std: null, n: 1, provenance: 'existing' }} format="integer" />
                <StatTile label="Best round" metric={{ mean: detail.best_round, std: null, n: 1, provenance: 'existing' }} format="integer" />
                <StatTile
                  label="Participation rate"
                  metric={{ mean: detail.config.participation_rate, std: null, n: 1, provenance: 'existing' }}
                  format="percent"
                />
                <StatTile
                  label="Elapsed"
                  metric={{ mean: detail.config.elapsed_seconds ? detail.config.elapsed_seconds / 60 : null, std: null, n: 1, provenance: 'existing', note: 'minutes' }}
                  digits={1}
                />
              </div>

              <div className="grid min-w-0 gap-4 xl:grid-cols-2">
                {['macro_f1', 'val_loss'].map((groupKey) => {
                  const seriesGroup = detail.series.filter((s) =>
                    groupKey === 'macro_f1' ? ['macro_f1', 'accuracy'].includes(s.key) : s.key === groupKey,
                  );
                  return (
                    <ChartFrame
                      key={groupKey}
                      title={groupKey === 'macro_f1' ? 'Classification per round' : 'Centralized validation loss/MSE'}
                      provenance="existing"
                    >
                      <SeriesLineChart series={seriesGroup.map((s) => ({ ...s, label: SERIES_LABELS[s.key] ?? s.label }))} />
                    </ChartFrame>
                  );
                })}
                {detail.series.some((s) => s.key === 'client_loss') && (
                  <ChartFrame title="Distributed client loss" provenance="existing">
                    <SeriesLineChart series={detail.series.filter((s) => s.key === 'client_loss')} />
                  </ChartFrame>
                )}
                {detail.series.some((s) => s.key === 'mb_per_round') && (
                  <ChartFrame title="Communication cost per round" provenance="derived">
                    <SeriesLineChart series={detail.series.filter((s) => s.key === 'mb_per_round')} />
                  </ChartFrame>
                )}
              </div>

              <Callout kind="mechanism" label="Convergence note">
                <p>{detail.convergence_note}</p>
              </Callout>

              <Card>
                <CardHeader
                  title={
                    <span className="flex items-center gap-2">
                      <Users size={15} className="text-accent" /> Client panel
                    </span>
                  }
                  subtitle={`${detail.clients.filter((c) => c.participated).length} of ${detail.clients.length} clients evaluated — participation and score only, never raw client data`}
                />
                <div className="max-h-96 overflow-y-auto">
                  <DataTable columns={clientColumns} rows={detail.clients} rowKey={(c) => String(c.client_id)} dense />
                </div>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}
