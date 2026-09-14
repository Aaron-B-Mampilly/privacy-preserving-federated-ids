import { Lock, Shield, ShieldAlert } from 'lucide-react';
import { api } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import {
  Badge,
  Callout,
  Card,
  CardHeader,
  DataTable,
  ErrorState,
  LoadingState,
  fmt,
} from '@/components/primitives';
import type { Column } from '@/components/primitives';
import { CategoryBarChart, ChartFrame } from '@/charts/ChartFrame';
import type { PrivacyResponse } from '@/types/api';

const EPS_LABEL: Record<string, string> = { inf: '∞', '8': '8', '3': '3', '1': '1', '0.5': '0.5' };

const MIA_LABELS: Record<string, string> = {
  fedavg: 'FedAvg updates',
  ours_dp: 'Ours + DP',
  ours_dp_secagg: 'Ours + DP + SecAgg+',
};

const GI_LABELS: Record<string, string> = {
  fedavg_no_protection: 'FedAvg (no protection)',
  ours_dp: 'Ours + DP',
  ours_dp_secagg: 'Ours + DP + SecAgg+',
};

export default function Privacy() {
  const { dataset, scope } = useDataset();
  const { data, error, loading, refresh } = useApi<PrivacyResponse>(
    () => api.privacy(dataset, scope),
    [dataset, scope],
  );

  if (loading && !data) return <LoadingState label="Loading privacy configuration" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const dp = data.differential_privacy;
  const sec = data.secure_aggregation;
  const attacks = data.attacks;

  const curveRows = data.utility_curve.filter((c) => c.status !== 'pending');
  const curveChart = curveRows.map((c) => ({
    name: `ε=${EPS_LABEL[c.epsilon] ?? c.epsilon}`,
    macro_f1: c.macro_f1?.mean ?? null,
    rare: c.rare_class_recall?.mean ?? null,
  }));

  const curveColumns: Column<(typeof curveRows)[number]>[] = [
    { key: 'eps', header: 'ε', render: (r) => <span className="font-mono font-medium">{EPS_LABEL[r.epsilon] ?? r.epsilon}</span> },
    { key: 'achieved', header: 'Achieved ε', align: 'right', render: (r) => <span>{fmt(r.achieved_epsilon, 3)}</span> },
    { key: 'f1', header: 'Macro-F1', align: 'right', render: (r) => <span>{fmt(r.macro_f1?.mean)} ± {fmt(r.macro_f1?.std)}</span> },
    { key: 'rare', header: 'Rare recall', align: 'right', render: (r) => <span>{fmt(r.rare_class_recall?.mean)}</span> },
  ];

  return (
    <div className="space-y-6">
      <header>
        <div className="flex items-center gap-2">
          <Shield size={22} className="text-accent" />
          <h1 className="font-display text-2xl">Privacy & Security</h1>
        </div>
        <p className="mt-1 max-w-2xl text-sm text-ink-muted">
          Differential privacy configuration, secure aggregation state, the privacy–utility curve,
          and empirical attack evaluation for {data.scope.label}.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader
            title={<span className="flex items-center gap-2"><Lock size={15} className="text-accent" /> Differential privacy</span>}
            action={<Badge tone={dp.enabled ? 'good' : 'neutral'}>{dp.enabled ? 'enabled' : 'not configured'}</Badge>}
          />
          {dp.enabled ? (
            <dl className="space-y-2 text-sm">
              <Row label="Mechanism" value={dp.mechanism} />
              <Row label="Clipping" value={dp.clipping} />
              <Row label="Noise" value={dp.noise_mechanism} />
              <Row label="Accountant" value={dp.accountant} />
              <Row label="Target ε" value={fmt(dp.target_epsilon, 2)} mono />
              <Row label="Achieved ε" value={fmt(dp.achieved_epsilon, 4)} mono />
              <Row label="Target δ" value={fmt(dp.target_delta, 8)} mono />
              <Row label="Noise multiplier" value={fmt(dp.noise_multiplier, 4)} mono />
              <Row label="Median clip norm" value={fmt(dp.median_clip_norm, 3)} mono />
            </dl>
          ) : (
            <p className="text-sm text-ink-muted">{dp.reason}</p>
          )}
        </Card>

        <Card>
          <CardHeader
            title={<span className="flex items-center gap-2"><ShieldAlert size={15} className="text-accent" /> Secure aggregation</span>}
          />
          <div className="flex flex-wrap gap-2">
            <Badge tone={sec.secagg_only_available ? 'good' : 'neutral'}>SecAgg+ only: {sec.secagg_only_available ? 'ready' : 'n/a'}</Badge>
            <Badge tone={sec.dp_secagg_available ? 'good' : 'neutral'}>DP + SecAgg+: {sec.dp_secagg_available ? 'ready' : 'n/a'}</Badge>
          </div>
          {sec.secagg_only && (
            <div className="mt-3 text-sm">
              <p className="label-caps mb-1">SecAgg+ alone</p>
              <p>Macro-F1 {fmt(sec.secagg_only.macro_f1?.mean)} · Accuracy {fmt(sec.secagg_only.accuracy?.mean)}</p>
            </div>
          )}
          {sec.dp_secagg && (
            <div className="mt-3 text-sm">
              <p className="label-caps mb-1">Combined with DP</p>
              <p>Macro-F1 {fmt(sec.dp_secagg.macro_f1?.mean)} · clip bound {fmt(sec.dp_secagg.clip_bound, 2)} · achieved ε {fmt(sec.dp_secagg.achieved_epsilon, 3)}</p>
              <p className="mt-1 text-xs text-ink-muted">{sec.dp_secagg.note}</p>
            </div>
          )}
        </Card>
      </div>

      <ChartFrame title="Privacy–utility curve" subtitle="Figure F1 — ε vs Macro-F1 and rare-class recall" provenance="existing">
        <CategoryBarChart
          data={curveChart}
          xKey="name"
          bars={[{ key: 'macro_f1', label: 'Macro-F1' }, { key: 'rare', label: 'Rare-class recall' }]}
        />
      </ChartFrame>
      <DataTable columns={curveColumns} rows={curveRows} rowKey={(r) => r.epsilon} />

      <Card>
        <CardHeader title="Empirical privacy evaluation" subtitle="Membership inference and gradient inversion" />
        {attacks.available ? (
          <div className="space-y-5">
            <div>
              <p className="label-caps mb-2">Membership inference (loss threshold)</p>
              <DataTable
                rowKey={(r: any) => r.key}
                rows={Object.entries(attacks.membership_inference ?? {}).map(([key, value]) => ({ key, label: MIA_LABELS[key] ?? key, value }))}
                columns={[
                  { key: 'label', header: 'Comparator', render: (r: any) => <span className="font-medium">{r.label}</span> },
                  { key: 'auc', header: 'AUC', align: 'right', render: (r: any) => <span>{fmt((r.value as any)?.auc, 3)}</span> },
                  { key: 'adv', header: 'Advantage', align: 'right', render: (r: any) => <span>{fmt((r.value as any)?.advantage, 3)}</span> },
                ]}
              />
            </div>
            <div>
              <p className="label-caps mb-2">Gradient inversion (DLG)</p>
              <DataTable
                rowKey={(r: any) => r.key}
                rows={Object.entries(attacks.gradient_inversion ?? {}).map(([key, value]) => ({ key, label: GI_LABELS[key] ?? key, value }))}
                columns={[
                  { key: 'label', header: 'Comparator', render: (r: any) => <span className="font-medium">{r.label}</span> },
                  {
                    key: 'mse', header: 'Reconstruction MSE', align: 'right',
                    render: (r: any) => {
                      const mse = r.value?.reconstruction_mse;
                      if (mse?.mean == null) return <Badge tone="good">diverged</Badge>;
                      return <span>{fmt(mse.mean, 2)}{mse.std != null && <span className="text-ink-muted"> ± {fmt(mse.std, 2)}</span>}</span>;
                    },
                  },
                  {
                    key: 'runs', header: 'Converged / total', align: 'right',
                    render: (r: any) => <span className="font-mono">
                      {((r.value as any)?.num_total_runs ?? 0) - ((r.value as any)?.num_diverged_runs ?? 0)}/{(r.value as any)?.num_total_runs ?? 0}
                    </span>,
                  },
                ]}
              />
            </div>
            <Callout kind="finding" label="Reading these numbers">
              <p>{attacks.reading_guide?.gradient_inversion}</p>
            </Callout>
          </div>
        ) : (
          <Callout kind="mechanism" label="Not run for this scope">
            <p>{attacks.reason}</p>
            {attacks.membership_inference_this_scope && (
              <p className="mt-2">
                This scope's own non-DP MIA AUC: {fmt(attacks.membership_inference_this_scope.non_dp?.auc, 3)}
              </p>
            )}
          </Callout>
        )}
      </Card>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-ink-muted">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : 'text-right text-sm'}>{value}</dd>
    </div>
  );
}
