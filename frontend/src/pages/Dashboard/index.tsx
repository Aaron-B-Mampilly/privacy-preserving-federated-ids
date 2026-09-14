import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Gauge,
  Radar,
  Shield,
  Target,
  Users,
  Wifi,
} from 'lucide-react';
import { api } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import {
  Badge,
  Callout,
  Card,
  CardHeader,
  ErrorState,
  LoadingState,
  StatTile,
  fmt,
} from '@/components/primitives';
import { ChartFrame, SeriesLineChart } from '@/charts/ChartFrame';
import type { DashboardResponse, EventItem } from '@/types/api';

const EVENT_META: Record<EventItem['kind'], { icon: typeof Activity; tone: string }> = {
  round: { icon: CheckCircle2, tone: 'text-good' },
  info: { icon: Activity, tone: 'text-ink-muted' },
  drift: { icon: Radar, tone: 'text-finding' },
  retrain: { icon: Cpu, tone: 'text-accent' },
  privacy: { icon: Shield, tone: 'text-caveat' },
  warning: { icon: AlertTriangle, tone: 'text-danger' },
};

const EPSILON_TICKS = ['∞', '8', '3', '1', '0.5'];

export default function Dashboard() {
  const { dataset, scope, datasetInfo } = useDataset();
  const { data, error, loading, refresh } = useApi<DashboardResponse>(
    () => api.dashboard(dataset, scope),
    [dataset, scope],
  );

  if (loading && !data) return <LoadingState label="Loading dashboard" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const trainingSeries = data.series.filter((s) => ['macro_f1', 'accuracy'].includes(s.key));
  const valSeries = data.series.filter((s) => s.key === 'val_mse');
  const commsSeries = data.series.filter((s) => s.key === 'mb_per_round');
  const privacySeries = data.series.find((s) => s.key === 'privacy_utility');

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl">FedPDA-IDS</h1>
        <p className="mt-1 text-sm text-ink-muted">
          Privacy-Preserving Federated Intrusion Detection · {data.scope.label}
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge tone="accent">{datasetInfo?.label ?? dataset}</Badge>
          <Badge>{data.scope.num_clients} clients</Badge>
          <Badge>{data.scope.clients_per_round}/round</Badge>
          <Badge>{data.scope.num_features} features</Badge>
          <span className="font-mono text-xs text-ink-muted">source: {data.source_run}</span>
        </div>
      </header>

      <section>
        <h2 className="label-caps mb-2">Primary metrics</h2>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
          <StatTile label="Macro-F1" metric={data.kpis.macro_f1} icon={<Target size={14} />} />
          <StatTile label="Rare-class recall" metric={data.kpis.rare_class_recall} icon={<Target size={14} />} />
          <StatTile label="Zero-day detection" metric={data.kpis.zero_day_detection} format="percent" icon={<Radar size={14} />} />
          <StatTile label="Privacy budget ε" metric={data.kpis.privacy_epsilon} digits={3} icon={<Shield size={14} />} />
          <StatTile label="FL round" metric={data.kpis.current_round} format="integer" icon={<Cpu size={14} />} />
          <StatTile label="Drift delay (rounds)" metric={data.kpis.drift_delay_rounds} format="integer" icon={<Radar size={14} />} />
        </div>
      </section>

      <section>
        <h2 className="label-caps mb-2">System</h2>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatTile label="Active clients" metric={data.kpis.active_clients} format="integer" icon={<Users size={14} />} />
          <StatTile label="Comms / round (MB)" metric={data.kpis.mb_per_round} digits={2} icon={<Wifi size={14} />} />
          <StatTile label="Convergence round" metric={data.kpis.convergence_round} format="integer" icon={<Gauge size={14} />} />
          <StatTile label="Accuracy" metric={data.kpis.accuracy} icon={<Target size={14} />} />
        </div>
      </section>

      <Callout kind="mechanism" label="How to read this page">
        <p>
          Federated training in this project runs offline in multi-hour batches, so the round-by-round
          charts below are a replay of a completed run's own recorded history — not a live trainer.
          Every value carries a provenance tag: <strong>stored</strong> results come from a completed
          experiment, <strong>derived</strong> values are computed from a saved confusion matrix or
          checkpoint with no new training.
        </p>
      </Callout>

      <div className="grid min-w-0 gap-4 xl:grid-cols-2">
        <ChartFrame
          title="Classification per round"
          subtitle="Distributed client evaluation, aggregated by Flower"
          provenance="existing"
        >
          <SeriesLineChart series={trainingSeries} />
        </ChartFrame>

        <ChartFrame
          title="Centralized validation MSE"
          subtitle="Reconstruction error on the held-out validation split"
          provenance="existing"
        >
          <SeriesLineChart series={valSeries} />
        </ChartFrame>

        <ChartFrame
          title="Privacy–utility (ε vs Macro-F1)"
          subtitle="Client-level DP sweep — x axis is ε: ∞, 8, 3, 1, 0.5"
          provenance="existing"
        >
          {privacySeries ? (
            <SeriesLineChart
              series={[{ ...privacySeries, label: 'Macro-F1' }]}
              xLabel="Privacy budget ε"
              xTicks={EPSILON_TICKS}
            />
          ) : (
            <div className="grid h-full place-items-center text-sm text-ink-muted">
              No DP sweep for this scope.
            </div>
          )}
        </ChartFrame>

        <ChartFrame
          title="Communication cost per round"
          subtitle="Both directions, all sampled clients — fixed by parameter shapes"
          provenance="derived"
        >
          <SeriesLineChart series={commsSeries} />
        </ChartFrame>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Recent events" subtitle="Reconstructed from completed run artifacts" />
          <ul className="space-y-3">
            {data.events.map((event, i) => {
              const meta = EVENT_META[event.kind] ?? EVENT_META.info;
              const Icon = meta.icon;
              return (
                <li key={`${event.title}-${i}`} className="flex gap-3">
                  <span className={`mt-0.5 shrink-0 ${meta.tone}`}>
                    <Icon size={15} />
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium">{event.title}</p>
                    {event.detail && <p className="text-xs text-ink-muted">{event.detail}</p>}
                  </div>
                  {event.round != null && (
                    <span className="ml-auto shrink-0 font-mono text-xs text-ink-muted">
                      r{event.round}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </Card>

        <Card>
          <CardHeader title="Model status" />
          <dl className="space-y-2 text-sm">
            {Object.entries(data.model_status).map(([key, value]) => (
              <div key={key} className="flex items-start justify-between gap-3">
                <dt className="text-ink-muted">{key.replace(/_/g, ' ')}</dt>
                <dd className="text-right font-mono text-xs">
                  {typeof value === 'boolean' ? (
                    <Badge tone={value ? 'good' : 'warning'}>{value ? 'yes' : 'no'}</Badge>
                  ) : (
                    String(value ?? '—')
                  )}
                </dd>
              </div>
            ))}
          </dl>
          {data.kpis.mb_per_round?.mean != null && (
            <p className="mt-3 border-t border-border pt-3 text-xs text-ink-muted">
              Each round exchanges {fmt(data.kpis.mb_per_round.mean, 2)} MB across{' '}
              {data.scope.clients_per_round} sampled clients (upload + broadcast).
            </p>
          )}
        </Card>
      </div>

      <p className="text-xs text-ink-muted">
        ε axis order: {EPSILON_TICKS.join(' → ')}. Datasets are never merged — every value on this page
        belongs to {data.scope.label}.
      </p>
    </div>
  );
}
