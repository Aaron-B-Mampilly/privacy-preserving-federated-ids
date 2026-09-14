import type { ReactNode } from 'react';
import { AlertTriangle, Inbox, Loader2, PlugZap, RefreshCw } from 'lucide-react';
import type { ApiError, MetricValue, Provenance } from '@/types/api';

/* ------------------------------------------------------------------ utils */

export function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ');
}

/** Formats a metric for display. `null` renders as an em-dash, never 0. */
export function fmt(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return '∞';
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return value.toFixed(digits);
}

export function fmtPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/* ------------------------------------------------------------------- Card */

export function Card({
  children,
  className,
  as: Tag = 'section',
}: {
  children: ReactNode;
  className?: string;
  as?: 'section' | 'div' | 'article';
}) {
  return <Tag className={cx('card p-5', className)}>{children}</Tag>;
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h3 className="text-base leading-tight">{title}</h3>
        {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------ Provenance */

const PROVENANCE_COPY: Record<Provenance, { label: string; title: string; className: string }> = {
  existing: {
    label: 'stored',
    title: 'Completed experiment result, reused as-is',
    className: 'bg-surface-alt text-ink-muted',
  },
  new: {
    label: 'new run',
    title: 'Experiment run during the final reporting pass',
    className: 'bg-accent-soft text-accent-strong',
  },
  derived: {
    label: 'derived',
    title: 'Computed from a saved confusion matrix or checkpoint — no new training',
    className: 'bg-finding-soft text-finding',
  },
  live: {
    label: 'live',
    title: 'Computed just now by this API process',
    className: 'bg-good-soft text-good',
  },
  demo: {
    label: 'demo data',
    title: 'Bundled fallback shown because the backend is unavailable — not a live result',
    className: 'bg-caveat-soft text-caveat',
  },
};

export function ProvenanceTag({ provenance }: { provenance: Provenance }) {
  const meta = PROVENANCE_COPY[provenance] ?? PROVENANCE_COPY.existing;
  return (
    <span
      title={meta.title}
      className={cx('rounded px-1.5 py-0.5 text-2xs font-semibold uppercase tracking-wide', meta.className)}
    >
      {meta.label}
    </span>
  );
}

/* --------------------------------------------------------------- StatTile */

export function StatTile({
  label,
  metric,
  format = 'number',
  digits = 4,
  hint,
  icon,
}: {
  label: string;
  metric: MetricValue | undefined;
  format?: 'number' | 'percent' | 'integer';
  digits?: number;
  hint?: string;
  icon?: ReactNode;
}) {
  const value = metric?.mean ?? null;
  const rendered =
    value === null
      ? '—'
      : format === 'percent'
        ? fmtPercent(value)
        : format === 'integer'
          ? String(Math.round(value))
          : fmt(value, digits);

  return (
    <div className="card p-4">
      <div className="flex items-start justify-between gap-2">
        <span className="label-caps">{label}</span>
        {icon && <span className="text-ink-muted/70">{icon}</span>}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="metric-value text-2xl leading-none">{rendered}</span>
        {metric?.std != null && (
          <span className="font-mono text-xs text-ink-muted">± {fmt(metric.std, digits)}</span>
        )}
      </div>
      <div className="mt-2 flex items-center gap-2">
        {metric && <ProvenanceTag provenance={metric.provenance} />}
        {metric?.n ? <span className="text-2xs text-ink-muted">n={metric.n}</span> : null}
      </div>
      {(metric?.note || hint) && (
        <p className="mt-2 text-xs leading-snug text-ink-muted">{metric?.note ?? hint}</p>
      )}
      {value === null && !metric?.note && (
        <p className="mt-2 text-xs text-ink-muted">Not measured for this scope.</p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ Badge */

type BadgeTone = 'neutral' | 'accent' | 'good' | 'warning' | 'danger';

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-surface-alt text-ink-muted border-border',
  accent: 'bg-accent-soft text-accent-strong border-accent/30',
  good: 'bg-good-soft text-good border-good/30',
  warning: 'bg-finding-soft text-finding border-finding/30',
  danger: 'bg-danger-soft text-danger border-danger/30',
};

export function Badge({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium',
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ---------------------------------------------------------------- Callout */

type CalloutKind = 'finding' | 'caveat' | 'mechanism' | 'resolved';

const CALLOUT_STYLES: Record<CalloutKind, { wrap: string; label: string }> = {
  finding: { wrap: 'bg-finding-soft border-finding/25', label: 'text-finding' },
  caveat: { wrap: 'bg-caveat-soft border-caveat/25', label: 'text-caveat' },
  mechanism: { wrap: 'bg-surface-alt border-border', label: 'text-ink-muted' },
  resolved: { wrap: 'bg-good-soft border-good/25', label: 'text-good' },
};

/** The same finding / caveat / mechanism / resolved vocabulary the
 *  project's published results artifact uses, so the dashboard and the
 *  write-up read as one system. */
export function Callout({
  kind = 'mechanism',
  label,
  children,
}: {
  kind?: CalloutKind;
  label: string;
  children: ReactNode;
}) {
  const style = CALLOUT_STYLES[kind];
  return (
    <div className={cx('rounded-card border p-4', style.wrap)}>
      <span className={cx('label-caps mb-2 block', style.label)}>{label}</span>
      <div className="space-y-2 text-sm leading-relaxed">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------ Load/Empty/Error */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cx('relative overflow-hidden rounded bg-surface-alt', className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/25 to-transparent" />
    </div>
  );
}

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-12 text-sm text-ink-muted">
      <Loader2 size={16} className="animate-spin" />
      {label}…
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  icon,
}: {
  title: string;
  detail?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <span className="text-ink-muted/60">{icon ?? <Inbox size={22} />}</span>
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="max-w-md text-sm text-ink-muted">{detail}</p>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="rounded-card border border-danger/30 bg-danger-soft p-5">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-danger">
          {error.offline ? <PlugZap size={18} /> : <AlertTriangle size={18} />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-danger">{error.error}</p>
          <p className="mt-1 text-sm text-ink">{error.detail}</p>
          {error.hint && <p className="mt-2 text-xs text-ink-muted">{error.hint}</p>}
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium hover:bg-surface-alt"
            >
              <RefreshCw size={12} /> Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- DataTable */

export interface Column<T> {
  key: string;
  header: string;
  align?: 'left' | 'right';
  render: (row: T) => ReactNode;
  width?: string;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  emptyMessage = 'No rows to display.',
  dense = false,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T, index: number) => string;
  emptyMessage?: string;
  dense?: boolean;
}) {
  if (!rows.length) {
    return <EmptyState title={emptyMessage} />;
  }
  return (
    <div className="overflow-x-auto rounded-card border border-border bg-surface">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                style={col.width ? { width: col.width } : undefined}
                className={cx(
                  'label-caps whitespace-nowrap border-b border-border bg-surface-alt px-3 py-2.5',
                  col.align === 'right' ? 'text-right' : 'text-left',
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey(row, index)} className="hover:bg-surface-alt/60">
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cx(
                    'whitespace-nowrap border-b border-border px-3',
                    dense ? 'py-1.5' : 'py-2.5',
                    col.align === 'right' ? 'text-right font-mono' : 'text-left',
                  )}
                >
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ----------------------------------------------------------------- Button */

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  type = 'button',
  className,
  icon,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  type?: 'button' | 'submit';
  className?: string;
  icon?: ReactNode;
}) {
  const styles = {
    primary: 'bg-accent text-white hover:bg-accent-strong border-transparent disabled:bg-accent/50',
    secondary: 'bg-surface border-border hover:bg-surface-alt',
    ghost: 'bg-transparent border-transparent hover:bg-surface-alt',
  }[variant];

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        'inline-flex items-center justify-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60',
        styles,
        className,
      )}
    >
      {icon}
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------- Tabs */

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: Array<{ id: T; label: string; sublabel?: string; disabled?: boolean }>;
  active: T;
  onChange: (id: T) => void;
}) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 border-b border-border">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={active === tab.id}
          disabled={tab.disabled}
          onClick={() => onChange(tab.id)}
          className={cx(
            '-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40',
            active === tab.id
              ? 'border-accent text-accent-strong'
              : 'border-transparent text-ink-muted hover:text-ink',
          )}
        >
          <span className="font-mono text-xs">{tab.id}</span>
          <span className="ml-2">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
