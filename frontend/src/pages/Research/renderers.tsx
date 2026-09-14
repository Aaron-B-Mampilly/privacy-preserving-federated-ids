import type { ReactNode } from 'react';
import { Badge, Callout, DataTable, EmptyState, fmt, fmtPercent } from '@/components/primitives';
import { CategoryBarChart } from '@/charts/ChartFrame';
import type { Column } from '@/components/primitives';

/* ------------------------------------------------------------------ utils */

interface Agg {
  mean: number | null;
  std: number | null;
  n: number;
}

function isPending(value: any): value is { status: 'pending'; reason: string } {
  return value && typeof value === 'object' && value.status === 'pending';
}

/** Renders mean±std, or the explicit reason a cell was never measured. */
export function Metric({ value, digits = 4 }: { value: any; digits?: number }) {
  if (isPending(value)) {
    return (
      <span title={value.reason} className="text-2xs uppercase tracking-wide text-ink-muted">
        not measured
      </span>
    );
  }
  if (value === null || value === undefined) return <span className="text-ink-muted">—</span>;
  if (typeof value === 'number') return <span>{fmt(value, digits)}</span>;
  if (typeof value === 'string') return <span>{value}</span>;

  const agg = value as Agg;
  if (agg.mean === null || agg.mean === undefined) {
    return <span className="text-2xs uppercase tracking-wide text-ink-muted">not measured</span>;
  }
  return (
    <span>
      {fmt(agg.mean, digits)}
      {agg.std != null && <span className="text-ink-muted"> ± {fmt(agg.std, digits)}</span>}
    </span>
  );
}

function meanOf(value: any): number | null {
  if (isPending(value) || value == null) return null;
  if (typeof value === 'number') return value;
  return typeof value.mean === 'number' ? value.mean : null;
}

/** Scope keys in the results file are snake_case identifiers
 *  ("cicids2017_alpha5"). Render them the way the datasets are actually
 *  named rather than title-casing the raw key. */
function prettyLabel(key: string): string {
  const alpha = key.match(/^cicids2017_alpha(.+)$/);
  if (alpha) return `CICIDS2017 α=${alpha[1]}`;
  if (key.startsWith('nbaiot_')) return `N-BaIoT ${key.replace('nbaiot_', '')}`;
  return key
    .replace(/_/g, ' ')
    .replace(/\beps\b/, 'ε')
    .replace(/^./, (c) => c.toUpperCase());
}

/** Charts inside the results explorer sit in the same card frame the
 *  tables do, so a section reads as one object rather than a table with
 *  a loose graphic under it. */
function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  // min-w-0 matters: a grid/flex child defaults to min-width:auto, which
  // lets Recharts' ResponsiveContainer resolve to a near-zero plot width
  // (bars vanish and the axis labels pile up in the corner).
  return (
    <div className="card min-w-0 p-5">
      <h4 className="label-caps mb-3">{title}</h4>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function mbOf(value: any): string {
  if (isPending(value)) return 'n/a';
  if (typeof value === 'string') return value;
  if (value && typeof value.mb_per_round === 'number') return fmt(value.mb_per_round, 2);
  return '—';
}

/* --------------------------------------------------------------- E1 / E2 */

const E1_ORDER = [
  ['local_only', 'Local-only'],
  ['centralized', 'Centralized'],
  ['fedavg', 'FedAvg'],
  ['fedprox', 'FedProx'],
  ['base_paper_replication', 'Base-paper replication'],
  ['ours_no_dp', 'Ours (no DP)'],
] as const;

export function E1Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const scopeData = data[scopeKey];
  if (!scopeData) return <EmptyState title="No data for this scope" />;

  const rows = E1_ORDER.map(([key, label]) => ({ key, label, entry: scopeData[key] ?? null }));

  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'comparator', header: 'Comparator', render: (r) => <span className="font-medium">{r.label}</span> },
    { key: 'accuracy', header: 'Accuracy', align: 'right', render: (r) => <Metric value={r.entry?.accuracy} /> },
    { key: 'macro_f1', header: 'Macro-F1', align: 'right', render: (r) => <Metric value={r.entry?.macro_f1} /> },
    { key: 'rare', header: 'Rare recall', align: 'right', render: (r) => <Metric value={r.entry?.rare_class_recall} /> },
    { key: 'fpr', header: 'FPR', align: 'right', render: (r) => <Metric value={r.entry?.fpr} digits={5} /> },
    {
      key: 'rounds',
      header: 'Rounds',
      align: 'right',
      render: (r) => {
        const v = r.entry?.rounds_to_convergence;
        if (isPending(v)) return <span className="text-2xs text-ink-muted">not measured</span>;
        return <span>{typeof v === 'number' ? v : String(v ?? '—').replace(/^n\/a.*/, 'n/a')}</span>;
      },
    },
    { key: 'mb', header: 'MB/round', align: 'right', render: (r) => <span>{mbOf(r.entry?.mb_per_round)}</span> },
  ];

  const chartData = rows
    .filter((r) => meanOf(r.entry?.macro_f1) !== null)
    .map((r) => ({ name: r.label, macro_f1: meanOf(r.entry?.macro_f1), rare: meanOf(r.entry?.rare_class_recall) }));

  return (
    <div className="space-y-4">
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.key} />
      <ChartCard title="Macro-F1 and rare-class recall by comparator">
        <CategoryBarChart
          data={chartData}
          xKey="name"
          xAngle={-25}
          bars={[
            { key: 'macro_f1', label: 'Macro-F1' },
            { key: 'rare', label: 'Rare-class recall' },
          ]}
        />
      </ChartCard>
    </div>
  );
}

export function E2Table({ data }: { data: any }) {
  const alphas = Object.keys(data);
  const mechanisms = [
    ['fedavg', 'FedAvg'],
    ['base_paper_replication', 'Base-paper replication'],
    ['ours', 'Ours'],
  ] as const;

  const rows = alphas.map((alpha) => ({ alpha, entry: data[alpha] }));

  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'alpha', header: 'α', render: (r) => <span className="font-mono font-medium">{r.alpha.replace('alpha_', '')}</span> },
    ...mechanisms.flatMap(([key, label]) => [
      {
        key: `${key}_f1`,
        header: `${label} · F1`,
        align: 'right' as const,
        render: (r: any) => <Metric value={r.entry?.[key]?.macro_f1} />,
      },
      {
        key: `${key}_rare`,
        header: `${label} · rare`,
        align: 'right' as const,
        render: (r: any) => <Metric value={r.entry?.[key]?.rare_class_recall} />,
      },
    ]),
  ];

  const chartData = rows.map((r) => ({
    name: `α=${r.alpha.replace('alpha_', '')}`,
    fedavg: meanOf(r.entry?.fedavg?.macro_f1),
    bpr: meanOf(r.entry?.base_paper_replication?.macro_f1),
    ours: meanOf(r.entry?.ours?.macro_f1),
  }));

  return (
    <div className="space-y-4">
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.alpha} />
      <ChartCard title="Macro-F1 across non-IID severity">
        <CategoryBarChart
          data={chartData}
          xKey="name"
          bars={[
            { key: 'fedavg', label: 'FedAvg' },
            { key: 'bpr', label: 'Base-paper replication' },
            { key: 'ours', label: 'Ours' },
          ]}
        />
      </ChartCard>
    </div>
  );
}

/* -------------------------------------------------------------------- E3 */

const EPS_ORDER = ['eps_inf', 'eps_8', 'eps_3', 'eps_1', 'eps_0.5'];
const EPS_LABEL: Record<string, string> = {
  eps_inf: '∞', 'eps_8': '8', 'eps_3': '3', 'eps_1': '1', 'eps_0.5': '0.5',
};

export function E3Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const scopeData = data[scopeKey];
  if (!scopeData) return <EmptyState title="No data for this scope" />;

  const rows = EPS_ORDER.filter((k) => scopeData[k]).map((k) => ({ eps: k, entry: scopeData[k] }));

  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'eps', header: 'ε', render: (r) => <span className="font-mono font-medium">{EPS_LABEL[r.eps]}</span> },
    { key: 'achieved', header: 'Achieved ε', align: 'right', render: (r) => <Metric value={r.entry.achieved_epsilon} digits={3} /> },
    { key: 'f1', header: 'Macro-F1', align: 'right', render: (r) => <Metric value={r.entry.macro_f1} /> },
    { key: 'rare', header: 'Rare-class recall', align: 'right', render: (r) => <Metric value={r.entry.rare_class_recall} /> },
  ];

  const chartData = rows.map((r) => ({
    name: `ε=${EPS_LABEL[r.eps]}`,
    macro_f1: meanOf(r.entry.macro_f1),
    rare: meanOf(r.entry.rare_class_recall),
  }));

  return (
    <div className="space-y-4">
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.eps} />
      <ChartCard title="Figure F1 — privacy–utility across the ε sweep">
        <CategoryBarChart
          data={chartData}
          xKey="name"
          bars={[
            { key: 'macro_f1', label: 'Macro-F1' },
            { key: 'rare', label: 'Rare-class recall' },
          ]}
        />
      </ChartCard>
      <Callout kind="finding" label="Figure F1 reading">
        <p>
          Macro-F1 declines gently across the sweep, but rare-class recall drops to exactly{' '}
          <strong>0.000</strong> at every finite ε — a measured zero, not a missing value. DP's cost
          here falls almost entirely on the rare attack classes.
        </p>
      </Callout>
    </div>
  );
}

/* -------------------------------------------------------------------- E4 */

export function E4Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const scopeData = data[scopeKey];
  if (!scopeData) return <EmptyState title="No data for this scope" />;

  const rows = EPS_ORDER.filter((k) => scopeData[k]).map((k) => ({ eps: k, entry: scopeData[k] }));

  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'eps', header: 'ε', render: (r) => <span className="font-mono font-medium">{EPS_LABEL[r.eps]}</span> },
    {
      key: 'a',
      header: '(a) unknown vs benign',
      align: 'right',
      render: (r) => <span>{fmtPercent(r.entry.a_unknown_vs_benign_detection_rate)}</span>,
    },
    {
      key: 'b',
      header: '(b) NEW CLASS',
      align: 'right',
      render: (r) => <span>{fmtPercent(r.entry.b_new_class_detection_rate)}</span>,
    },
    {
      key: 'fp',
      header: 'FP on known',
      align: 'right',
      render: (r) => <span>{fmtPercent(r.entry.false_positive_rate_on_known)}</span>,
    },
    { key: 'c', header: '(c) known-class F1', align: 'right', render: (r) => <Metric value={r.entry.c_known_class_f1} /> },
  ];

  const chartData = rows.map((r) => ({
    name: `ε=${EPS_LABEL[r.eps]}`,
    a: r.entry.a_unknown_vs_benign_detection_rate,
    b: r.entry.b_new_class_detection_rate,
    fp: r.entry.false_positive_rate_on_known,
  }));

  return (
    <div className="space-y-4">
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.eps} />
      <ChartCard title="Zero-day detection criteria across ε">
        <CategoryBarChart
          data={chartData}
          xKey="name"
          bars={[
            { key: 'a', label: '(a) unknown vs benign' },
            { key: 'b', label: '(b) NEW CLASS' },
            { key: 'fp', label: 'False positives on known' },
          ]}
        />
      </ChartCard>
      <Callout kind="caveat" label="(b) at 100% is not success">
        <p>
          Where (b) reads 1.000 under DP, the mechanism is flagging <em>everything</em> as a new class —
          note the false-positive column rising with it. That is a saturation failure, not detection.
        </p>
      </Callout>
    </div>
  );
}

/* -------------------------------------------------------------------- E5 */

export function E5Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const entry = data[scopeKey];
  if (!entry || isPending(entry)) return <EmptyState title="No drift/recovery run for this scope" />;

  const matched = entry.matched_subset_comparison ?? {};
  const isMatched = matched.num_clients_matched === matched.num_clients_total;

  const rows = [
    { stage: 'F1 before drift', full: entry.f1_before_drift?.macro_f1_mean, sub: matched.f1_before_drift },
    { stage: 'F1 after drift (no adaptation)', full: entry.f1_after_drift_no_adaptation_without_monitor?.macro_f1_mean, sub: matched.f1_after_drift_no_adaptation },
    { stage: 'F1 after triggered retraining', full: entry.f1_after_triggered_retraining_with_monitor?.macro_f1_mean, sub: matched.f1_after_triggered_retraining },
  ];

  const columns: Column<(typeof rows)[number]>[] = [
    { key: 'stage', header: 'Stage', render: (r) => <span className="font-medium">{r.stage}</span> },
    { key: 'full', header: 'Full pool', align: 'right', render: (r) => <Metric value={r.full} /> },
    { key: 'matched', header: `Matched subset (n=${matched.num_clients_matched ?? '—'})`, align: 'right', render: (r) => <Metric value={r.sub} /> },
  ];

  const chartData = rows.map((r) => ({ name: r.stage.replace('F1 ', ''), matched: meanOf(r.sub) }));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Badge tone="accent">Detection delay: {entry.drift_detection_delay_rounds} round(s)</Badge>
        <Badge>Retrain rounds: {entry.num_retrain_rounds_run}</Badge>
        <Badge tone={isMatched ? 'good' : 'warning'}>
          {matched.num_clients_matched}/{matched.num_clients_total} clients matched
        </Badge>
      </div>

      <DataTable columns={columns} rows={rows} rowKey={(r) => r.stage} />
      <ChartCard title="Drift and recovery (matched client subset)">
        <CategoryBarChart data={chartData} xKey="name" bars={[{ key: 'matched', label: 'Macro-F1 (matched subset)' }]} />
      </ChartCard>

      {!isMatched && (
        <Callout kind="caveat" label="Why the matched column is the one to read">
          <p>{matched.note}</p>
        </Callout>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- E6 */

const MIA_LABELS: Record<string, string> = {
  fedavg: 'FedAvg updates',
  ours_dp: 'Ours + DP',
  ours_dp_secagg: 'Ours + DP + SecAgg+',
};

const GI_LABELS: Record<string, string> = {
  fedavg_no_protection: 'FedAvg updates (no protection)',
  ours_dp: 'Ours + DP',
  ours_dp_secagg: 'Ours + DP + SecAgg+',
};

export function E6Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const entry = data[scopeKey];
  if (!entry) return <EmptyState title="Attack evaluation was not run for this scope" />;

  const miaRows = Object.entries(entry.mia ?? {}).map(([key, value]: [string, any]) => ({
    key, label: MIA_LABELS[key] ?? key, value,
  }));
  const giRows = Object.entries(entry.gradient_inversion ?? {}).map(([key, value]: [string, any]) => ({
    key, label: GI_LABELS[key] ?? key, value,
  }));

  return (
    <div className="space-y-5">
      <div>
        <h4 className="label-caps mb-2">Membership inference (loss threshold)</h4>
        <DataTable
          rowKey={(r) => r.key}
          rows={miaRows}
          columns={[
            { key: 'label', header: 'Comparator', render: (r) => <span className="font-medium">{r.label}</span> },
            { key: 'auc', header: 'AUC', align: 'right', render: (r) => <Metric value={r.value?.auc} digits={3} /> },
            { key: 'adv', header: 'Advantage', align: 'right', render: (r) => <Metric value={r.value?.advantage} digits={3} /> },
          ]}
        />
        <p className="mt-2 text-xs text-ink-muted">AUC 0.5 is chance — higher means the attack separates members better.</p>
      </div>

      <div>
        <h4 className="label-caps mb-2">Gradient inversion (DLG)</h4>
        <DataTable
          rowKey={(r) => r.key}
          rows={giRows}
          columns={[
            { key: 'label', header: 'Comparator', render: (r) => <span className="font-medium">{r.label}</span> },
            {
              key: 'mse',
              header: 'Reconstruction MSE',
              align: 'right',
              render: (r) =>
                r.value?.diverged && !r.value?.reconstruction_mse ? (
                  <Badge tone="good">diverged — attack failed</Badge>
                ) : (
                  <Metric value={r.value?.reconstruction_mse} digits={2} />
                ),
            },
            {
              key: 'runs',
              header: 'Converged / total',
              align: 'right',
              render: (r) => (
                <span className="font-mono">
                  {(r.value?.num_total_runs ?? 0) - (r.value?.num_diverged_runs ?? 0)}/{r.value?.num_total_runs ?? 0}
                </span>
              ),
            },
          ]}
        />
      </div>

      <Callout kind="finding" label="Reading the reconstruction column">
        <p>
          Reconstruction MSE is a privacy <strong>risk</strong> score: lower means more leaked. The
          unprotected FedAvg surface reconstructs real client data almost exactly; Ours+DP never
          converged in any run, and Ours+DP+SecAgg reconstructs far worse when it does. Diverged runs
          are excluded from the mean rather than folded in as enormous errors.
        </p>
      </Callout>
    </div>
  );
}

/* -------------------------------------------------------------------- T7 */

export function T7Table({ data, scopeKey }: { data: any; scopeKey: string }) {
  const rows: any[] = data[scopeKey] ?? [];
  if (!rows.length) return <EmptyState title="No ablation rows for this scope" />;

  const columns: Column<any>[] = [
    { key: 'row', header: 'Configuration', render: (r) => <span className="font-medium">{r.row}</span> },
    { key: 'eps', header: 'ε', align: 'right', render: (r) => <span className="font-mono">{r.epsilon === 'inf' ? '∞' : r.epsilon}</span> },
    { key: 'f1', header: 'Macro-F1', align: 'right', render: (r) => <Metric value={r.macro_f1?.macro_f1_mean ?? r.macro_f1} /> },
    { key: 'rare', header: 'Rare recall', align: 'right', render: (r) => <Metric value={r.rare_class_recall} /> },
    {
      key: 'zd',
      header: 'Zero-day (b)',
      align: 'right',
      render: (r) => {
        const v = r.zero_day_b_new_class_detection;
        if (v && typeof v === 'object' && v.status === 'not_applicable') {
          return <span title={v.reason} className="text-2xs uppercase text-ink-muted">n/a</span>;
        }
        return <span>{fmtPercent(typeof v === 'number' ? v : null)}</span>;
      },
    },
    { key: 'mb', header: 'MB/round', align: 'right', render: (r) => <span>{mbOf(r.mb_per_round)}</span> },
  ];

  const chartData = rows.map((r) => ({
    name: r.row.replace('+ ', '+'),
    macro_f1: r.macro_f1?.macro_f1_mean ?? meanOf(r.macro_f1),
    zero_day: typeof r.zero_day_b_new_class_detection === 'number' ? r.zero_day_b_new_class_detection : null,
  }));

  return (
    <div className="space-y-4">
      <DataTable columns={columns} rows={rows} rowKey={(r) => r.row} />
      <div className="grid min-w-0 gap-4 xl:grid-cols-2">
        <ChartCard title="Macro-F1 by module">
          <CategoryBarChart data={chartData} xKey="name" xAngle={-25} bars={[{ key: 'macro_f1', label: 'Macro-F1' }]} />
        </ChartCard>
        <ChartCard title="Zero-day NEW CLASS detection by module">
          <CategoryBarChart data={chartData} xKey="name" xAngle={-25} bars={[{ key: 'zero_day', label: 'Zero-day (b)' }]} />
        </ChartCard>
      </div>
      <Callout kind="finding" label="Incremental contribution">
        <p>
          Macro-F1 is flat across every row — on this scope the model has no class discrimination to
          lose. The module-by-module signal lives in the zero-day column instead.
        </p>
      </Callout>
    </div>
  );
}

export { prettyLabel };
