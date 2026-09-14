import type { ReactNode } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, CardHeader, EmptyState, ProvenanceTag, fmt } from '@/components/primitives';
import type { ChartSeries, Provenance } from '@/types/api';

/**
 * Chart colours are drawn from the design tokens rather than Recharts'
 * defaults, so charts, tables and badges stay one visual system in both
 * themes. Reading the computed CSS variable keeps dark mode correct
 * without duplicating the palette in JS.
 */
export function token(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value ? `rgb(${value})` : fallback;
}

export const chartPalette = () => [
  token('--accent', 'rgb(13,109,103)'),
  token('--finding', 'rgb(163,78,14)'),
  token('--caveat', 'rgb(138,59,82)'),
  token('--good', 'rgb(26,122,76)'),
  token('--ink-muted', 'rgb(92,100,120)'),
];

export function axisStyle() {
  return {
    stroke: token('--border', '#dde1e8'),
    tick: { fill: token('--ink-muted', '#5c6478'), fontSize: 11, fontFamily: 'IBM Plex Mono' },
  };
}

function TooltipContent({ active, payload, label, unit }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2 text-xs shadow-lift">
      <p className="mb-1 font-mono text-ink-muted">{label}</p>
      {payload.map((entry: any) => (
        <p key={entry.dataKey} className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-sm" style={{ background: entry.color }} />
          <span className="text-ink-muted">{entry.name}</span>
          <span className="ml-auto font-mono font-medium">
            {fmt(entry.value)}
            {unit ? ` ${unit}` : ''}
          </span>
        </p>
      ))}
    </div>
  );
}

export function ChartFrame({
  title,
  subtitle,
  provenance,
  children,
  action,
  height = 240,
}: {
  title: string;
  subtitle?: string;
  provenance?: Provenance;
  children: ReactNode;
  action?: ReactNode;
  height?: number;
}) {
  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            {title}
            {provenance && <ProvenanceTag provenance={provenance} />}
          </span>
        }
        subtitle={subtitle}
        action={action}
      />
      <div className="min-w-0" style={{ height }}>{children}</div>
    </Card>
  );
}

/** A round-indexed line chart built from one or more API series. */
export function SeriesLineChart({
  series,
  xLabel = 'Round',
  height = 240,
  xTicks,
}: {
  series: ChartSeries[];
  xLabel?: string;
  height?: number;
  /** Maps an x value to a display label -- used by the privacy-utility
   *  chart, whose x axis is an index into the epsilon sweep rather than
   *  a round number. Without it the axis would read 0..4. */
  xTicks?: string[];
}) {
  const present = series.filter((s) => s.points.length > 0);
  if (!present.length) {
    return <EmptyState title="No series recorded" detail="This run did not record these values." />;
  }

  const rounds = new Map<number, Record<string, number | null>>();
  present.forEach((s) => {
    s.points.forEach((p) => {
      const row = rounds.get(p.round) ?? { round: p.round };
      row[s.key] = p.value;
      rounds.set(p.round, row);
    });
  });
  const data = Array.from(rounds.values()).sort((a, b) => (a.round as number) - (b.round as number));
  const colors = chartPalette();
  const axis = axisStyle();

  const multi = present.length > 1;

  return (
    <ResponsiveContainer width="100%" height={height}>
      {/* The legend sits at the TOP: at the bottom it overlaps the x-axis
          label, and a dashboard reads better with the series named before
          the reader reaches the plot. */}
      <LineChart data={data} margin={{ top: multi ? 4 : 8, right: 12, bottom: 18, left: 4 }}>
        <CartesianGrid stroke={axis.stroke} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="round"
          stroke={axis.stroke}
          tick={axis.tick}
          tickFormatter={xTicks ? (v: number) => xTicks[v] ?? String(v) : undefined}
          label={{
            value: xLabel,
            position: 'insideBottom',
            offset: -8,
            fill: axis.tick.fill,
            fontSize: 11,
          }}
        />
        <YAxis stroke={axis.stroke} tick={axis.tick} width={56} />
        <Tooltip content={<TooltipContent unit={present[0]?.unit} />} />
        {multi && (
          <Legend
            verticalAlign="top"
            align="right"
            height={24}
            wrapperStyle={{ fontSize: 11, color: axis.tick.fill }}
            iconType="plainline"
          />
        )}
        {present.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={colors[i % colors.length]}
            strokeWidth={1.8}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Categorical comparison, used for the T7 ablation and E1/E2 tables. */
export function CategoryBarChart({
  data,
  bars,
  xKey,
  height = 260,
  xAngle = 0,
}: {
  data: Array<Record<string, any>>;
  bars: Array<{ key: string; label: string }>;
  xKey: string;
  height?: number;
  xAngle?: number;
}) {
  if (!data.length) return <EmptyState title="Nothing to compare yet" />;
  const colors = chartPalette();
  const axis = axisStyle();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: xAngle ? 56 : 18, left: 4 }}>
        <CartesianGrid stroke={axis.stroke} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={axis.stroke}
          tick={{ ...axis.tick, fontSize: 10 }}
          angle={xAngle}
          textAnchor={xAngle ? 'end' : 'middle'}
          interval={0}
          height={xAngle ? 60 : 30}
        />
        <YAxis stroke={axis.stroke} tick={axis.tick} width={56} />
        <Tooltip content={<TooltipContent />} cursor={{ fill: 'rgb(var(--surface-alt) / 0.6)' }} />
        {bars.length > 1 && (
          <Legend verticalAlign="top" align="right" height={24} wrapperStyle={{ fontSize: 11, color: axis.tick.fill }} />
        )}
        {bars.map((bar, i) => (
          <Bar
            key={bar.key}
            dataKey={bar.key}
            name={bar.label}
            fill={colors[i % colors.length]}
            radius={[3, 3, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
