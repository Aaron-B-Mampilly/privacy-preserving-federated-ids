import { useState } from 'react';
import { PlayCircle, Radar } from 'lucide-react';
import { api, toApiError } from '@/services/api';
import { useApi, usePolling } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import {
  Badge,
  Button,
  Callout,
  Card,
  CardHeader,
  ErrorState,
  LoadingState,
  fmt,
} from '@/components/primitives';
import { ChartFrame, axisStyle, token } from '@/charts/ChartFrame';
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { ApiError, DriftStatusResponse, RetrainJob } from '@/types/api';
import StateTimeline from './StateTimeline';

function MetricPair({ label, before, after, unit = '' }: { label: string; before: number | null; after: number | null; unit?: string }) {
  const delta = before != null && after != null ? after - before : null;
  return (
    <div className="rounded-md border border-border bg-surface-alt/50 p-3">
      <p className="text-2xs text-ink-muted">{label}</p>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="font-mono text-sm text-ink-muted line-through">{fmt(before)}{unit}</span>
        <span className="font-mono text-base font-semibold">{fmt(after)}{unit}</span>
      </div>
      {delta != null && (
        <p className={`mt-0.5 text-2xs font-medium ${delta >= 0 ? 'text-good' : 'text-danger'}`}>
          {delta >= 0 ? '+' : ''}{fmt(delta, 3)}
        </p>
      )}
    </div>
  );
}

export default function Drift() {
  const { dataset, scope } = useDataset();
  const { data, error, loading, refresh } = useApi<DriftStatusResponse>(
    () => api.driftStatus(dataset, scope),
    [dataset, scope],
  );

  const [job, setJob] = useState<RetrainJob | null>(null);
  const [rounds, setRounds] = useState(5);
  const [triggerError, setTriggerError] = useState<ApiError | null>(null);

  const jobActive = job != null && (job.state === 'queued' || job.state === 'running');
  usePolling(
    async () => {
      if (!job) return;
      const updated = await api.retrainStatus(job.job_id);
      setJob(updated);
      if (updated.state === 'completed' || updated.state === 'failed') refresh();
    },
    3000,
    jobActive,
  );

  async function triggerRetrain() {
    setTriggerError(null);
    try {
      const created = await api.triggerRetrain(dataset, scope, rounds);
      setJob(created);
    } catch (err) {
      setTriggerError(toApiError(err));
    }
  }

  if (loading && !data) return <LoadingState label="Loading drift monitor" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const matched = data.recovery.matched_subset;
  const detector = data.detector;
  const triggerRound: number | null = detector.first_trigger_round;

  const chartData = data.rounds.map((r) => ({ round: r.round, psi: r.psi, threshold: detector.psi_threshold }));

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Radar size={22} className="text-accent" />
          <h1 className="font-display text-2xl">Drift Monitor</h1>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Unsupervised concept-drift detection (ADWIN primary, PSI backup) and measured recovery after
          retraining, for {data.scope.label}.
        </p>
      </header>

      <Card>
        <StateTimeline state={data.state} />
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Detector configuration"
            action={triggerRound != null ? <Badge tone="warning">Trigger at round {triggerRound}</Badge> : <Badge tone="good">No trigger yet</Badge>}
          />
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <Row label="Primary" value={detector.primary} />
            <Row label="Backup" value={detector.backup} />
            <Row label="ADWIN δ" value={fmt(detector.adwin_delta, 4)} />
            <Row label="PSI threshold" value={fmt(detector.psi_threshold, 3)} />
            <Row label="Round window" value={String(detector.round_window_size)} />
            <Row label="Consecutive rounds for trigger" value={String(detector.consecutive_rounds_for_retrain)} />
            <Row label="ADWIN change points" value={String(detector.num_adwin_drift_points)} />
            <Row label="Rounds monitored" value={String(detector.num_rounds)} />
          </dl>
        </Card>

        <Card>
          <CardHeader title="Reference vs stream" subtitle="Reconstruction MSE" />
          <dl className="space-y-2 text-sm">
            <Row label="Reference mean" value={fmt(detector.reference_reconstruction_mse?.mean, 5)} />
            <Row label="Reference std" value={fmt(detector.reference_reconstruction_mse?.std, 5)} />
            <Row label="Stream mean" value={fmt(detector.stream_reconstruction_mse?.mean, 5)} />
            <Row label="Stream std" value={fmt(detector.stream_reconstruction_mse?.std, 5)} />
          </dl>
        </Card>
      </div>

      <ChartFrame title="PSI per round" subtitle="Backup detector — dashed line is the threshold" provenance="existing">
        <PsiChart data={chartData} threshold={detector.psi_threshold} />
      </ChartFrame>

      <Card>
        <CardHeader
          title="Trigger a retrain"
          subtitle="Real pipeline, always written to a sandboxed run name — never an official checkpoint"
        />
        {triggerRound == null ? (
          <Callout kind="mechanism" label="Nothing to retrain from">
            <p>No drift trigger has fired for this scope, so there is no cutoff point to retrain from.</p>
          </Callout>
        ) : (
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-sm">
              Retrain rounds
              <input
                type="number"
                min={1}
                max={20}
                value={rounds}
                onChange={(e) => setRounds(Number(e.target.value))}
                className="w-16 rounded-md border border-border bg-surface px-2 py-1 text-sm"
                disabled={jobActive}
              />
            </label>
            <Button variant="primary" icon={<PlayCircle size={14} />} disabled={jobActive} onClick={triggerRetrain}>
              {jobActive ? 'Running…' : 'Trigger retrain'}
            </Button>
            {job && (
              <Badge tone={job.state === 'completed' ? 'good' : job.state === 'failed' ? 'danger' : 'accent'}>
                {job.state} — {job.run_name}
              </Badge>
            )}
          </div>
        )}
        {triggerError && <div className="mt-3"><ErrorState error={triggerError} /></div>}
        {job?.message && <p className="mt-2 text-xs text-ink-muted">{job.message}</p>}

        {job?.state === 'completed' && job.result && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <MetricPair
              label="Macro-F1 (matched clients, held-out post-cutoff slice)"
              before={job.result.macro_f1_before_retrain}
              after={job.result.macro_f1_after_retrain}
            />
            <div className="rounded-md border border-border bg-surface-alt/50 p-3">
              <p className="text-2xs text-ink-muted">Clients retrained</p>
              <p className="mt-1 font-mono text-base font-semibold">{job.result.num_clients_retrained}</p>
              <p className="mt-1 text-2xs text-ink-muted">{job.result.evaluation_note}</p>
            </div>
          </div>
        )}
      </Card>

      {data.recovery.available ? (
        <Card>
          <CardHeader title="Recovery (from the last completed retrain)" />
          <div className="mb-3 flex flex-wrap gap-2">
            <Badge tone="accent">Delay: {data.recovery.detection_delay_rounds} round(s)</Badge>
            <Badge tone={matched?.num_clients_matched === matched?.num_clients_total ? 'good' : 'warning'}>
              {matched?.num_clients_matched}/{matched?.num_clients_total} clients matched
            </Badge>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <MetricPair label="Before drift" before={null} after={matched?.f1_before_drift?.mean} />
            <MetricPair label="After drift, no adaptation" before={matched?.f1_before_drift?.mean} after={matched?.f1_after_drift_no_adaptation?.mean} />
            <MetricPair label="After triggered retraining" before={matched?.f1_after_drift_no_adaptation?.mean} after={matched?.f1_after_triggered_retraining?.mean} />
          </div>
          {matched?.note && (
            <Callout kind="caveat" label="Why the matched subset matters">
              <p>{matched.note}</p>
            </Callout>
          )}
        </Card>
      ) : (
        <Callout kind="mechanism" label="No recovery data yet">
          <p>{data.recovery.reason}</p>
        </Callout>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-ink-muted">{label}</dt>
      <dd className="font-mono text-xs">{value}</dd>
    </div>
  );
}

/** Colours must go through token()/axisStyle() rather than a raw
 *  `var(--accent)` string: the design tokens are RGB TRIPLES ("13 109
 *  103"), not full colours, so an unwrapped var() reference resolves to
 *  an invalid SVG paint and the line silently fails to render. Found by
 *  visually checking this exact chart, not assumed. */
function PsiChart({ data, threshold }: { data: Array<{ round: number; psi: number | null; threshold: number }>; threshold: number }) {
  const axis = axisStyle();
  const accent = token('--accent', '#0d6d67');
  const finding = token('--finding', '#a34e0e');

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
        <CartesianGrid stroke={axis.stroke} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="round" stroke={axis.stroke} tick={axis.tick} label={{ value: 'Round', position: 'insideBottom', offset: -8, fill: axis.tick.fill, fontSize: 11 }} />
        <YAxis stroke={axis.stroke} tick={axis.tick} width={50} />
        <Tooltip />
        <ReferenceLine y={threshold} stroke={finding} strokeDasharray="4 4" />
        <Line type="monotone" dataKey="psi" stroke={accent} strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}
