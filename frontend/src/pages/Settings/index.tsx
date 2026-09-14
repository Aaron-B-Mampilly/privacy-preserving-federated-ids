import { Activity, CheckCircle2, Database, ExternalLink, Server, XCircle } from 'lucide-react';
import { API_BASE_URL, GRAFANA_URL } from '@/services/api';
import { useDataset } from '@/hooks/useDataset';
import { Badge, Callout, Card, CardHeader } from '@/components/primitives';

export default function Settings() {
  const { health, connection, datasets, dataset, scope, datasetInfo } = useDataset();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl">Settings</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Connection, active scope, and links to the observability stack.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <Server size={15} className="text-accent" /> Backend
              </span>
            }
            subtitle="Configured with VITE_API_BASE"
          />
          <dl className="space-y-2.5 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="text-ink-muted">API base</dt>
              <dd className="font-mono text-xs">{API_BASE_URL}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-ink-muted">Status</dt>
              <dd>
                <Badge tone={connection === 'connected' ? 'good' : connection === 'connecting' ? 'warning' : 'danger'}>
                  {connection === 'connected' ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                  {connection}
                </Badge>
              </dd>
            </div>
            {health && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-ink-muted">API version</dt>
                  <dd className="font-mono text-xs">{health.version}</dd>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-ink-muted">Result files</dt>
                  <dd className="font-mono text-xs">{health.results_available}</dd>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-ink-muted">Checkpoints</dt>
                  <dd className="font-mono text-xs">{health.checkpoints_available}</dd>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-ink-muted">Consolidated tables</dt>
                  <dd>
                    <Badge tone={health.consolidated_tables ? 'good' : 'warning'}>
                      {health.consolidated_tables ? 'present' : 'missing'}
                    </Badge>
                  </dd>
                </div>
              </>
            )}
          </dl>

          {health?.notes?.length ? (
            <div className="mt-4">
              <Callout kind="caveat" label="Notes">
                <ul className="space-y-1">
                  {health.notes.map((n) => (
                    <li key={n} className="text-sm">{n}</li>
                  ))}
                </ul>
              </Callout>
            </div>
          ) : null}
        </Card>

        <Card>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <Database size={15} className="text-accent" /> Active scope
              </span>
            }
            subtitle="Switched from the header. The two federations are never merged."
          />
          <dl className="space-y-2.5 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="text-ink-muted">Dataset</dt>
              <dd className="font-medium">{datasetInfo?.label ?? dataset}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-ink-muted">{datasetInfo?.scope_label ?? 'Scope'}</dt>
              <dd className="font-mono text-xs">{scope}</dd>
            </div>
            {datasetInfo && (
              <>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-ink-muted">Rare classes</dt>
                  <dd className="text-right text-xs">{datasetInfo.rare_labels.join(', ')}</dd>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-ink-muted">Zero-day holdout</dt>
                  <dd className="text-right text-xs">{datasetInfo.zero_day_holdout.join(', ')}</dd>
                </div>
              </>
            )}
          </dl>

          <div className="mt-4 space-y-2">
            {datasets.map((d) => (
              <div key={d.id} className="flex items-center justify-between rounded-md border border-border px-3 py-2">
                <div className="min-w-0">
                  <p className="text-sm font-medium">{d.label}</p>
                  <p className="truncate text-xs text-ink-muted">{d.description}</p>
                </div>
                <Badge tone={d.available ? 'good' : 'warning'}>{d.available ? 'ready' : 'no data'}</Badge>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Activity size={15} className="text-accent" /> Observability
            </span>
          }
          subtitle="Prometheus and Grafana remain the technical monitoring layer — this app does not replace them."
        />
        <div className="flex flex-wrap gap-3">
          <a
            href={health?.grafana_url ?? GRAFANA_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium hover:bg-surface-alt"
          >
            Open Grafana <ExternalLink size={13} />
          </a>
          {health?.prometheus_url && (
            <a
              href={health.prometheus_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium hover:bg-surface-alt"
            >
              Open Prometheus <ExternalLink size={13} />
            </a>
          )}
          <a
            href={`${API_BASE_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium hover:bg-surface-alt"
          >
            API docs <ExternalLink size={13} />
          </a>
        </div>
        <p className="mt-3 text-xs text-ink-muted">
          Prometheus metrics are exported on port 8000 only while a training run is executing, which
          is why this API runs on 8001.
        </p>
      </Card>
    </div>
  );
}
