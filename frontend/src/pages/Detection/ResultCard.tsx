import { AlertOctagon, CheckCircle2, HelpCircle, ShieldAlert, ShieldCheck } from 'lucide-react';
import { Badge, Card, ProvenanceTag, cx, fmt, fmtPercent } from '@/components/primitives';
import type { PredictionResponse } from '@/types/api';

const THREAT_META: Record<
  PredictionResponse['threat_level'],
  { label: string; icon: typeof ShieldCheck; wrap: string; icon_c: string }
> = {
  benign: { label: 'BENIGN', icon: ShieldCheck, wrap: 'bg-good-soft border-good/30', icon_c: 'text-good' },
  low: { label: 'LOW THREAT', icon: HelpCircle, wrap: 'bg-surface-alt border-border', icon_c: 'text-ink-muted' },
  medium: { label: 'KNOWN ATTACK', icon: ShieldAlert, wrap: 'bg-finding-soft border-finding/30', icon_c: 'text-finding' },
  high: { label: 'KNOWN ATTACK', icon: AlertOctagon, wrap: 'bg-danger-soft border-danger/30', icon_c: 'text-danger' },
  unknown: { label: 'UNKNOWN', icon: HelpCircle, wrap: 'bg-surface-alt border-border', icon_c: 'text-ink-muted' },
};

/** The verdict banner. NEW CLASS overrides the normal threat-level
 *  colouring — the prototype mechanism identifying something the
 *  classifier itself is confident about is the more important signal
 *  to surface, per E4/E7's zero-day framing. */
export default function ResultCard({ result }: { result: PredictionResponse }) {
  const meta = result.is_new_class
    ? { label: 'NEW CLASS DETECTED', icon: AlertOctagon, wrap: 'bg-caveat-soft border-caveat/30', icon_c: 'text-caveat' }
    : THREAT_META[result.threat_level];
  const Icon = meta.icon;

  return (
    <Card className="space-y-4">
      <div className={cx('flex items-center gap-3 rounded-card border p-4', meta.wrap)}>
        <Icon size={26} className={cx('shrink-0', meta.icon_c)} />
        <div className="min-w-0">
          <p className={cx('font-display text-lg font-semibold', meta.icon_c)}>{meta.label}</p>
          <p className="text-sm text-ink-muted">
            {result.is_new_class ? 'Not close to any known class prototype' : result.predicted_label}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="metric-value text-2xl">{fmtPercent(result.confidence, 1)}</p>
          <p className="text-2xs text-ink-muted">confidence</p>
        </div>
      </div>

      {result.true_label && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-ink-muted">Ground truth:</span>
          <Badge tone={result.true_label === result.predicted_label ? 'good' : 'warning'}>
            {result.true_label === result.predicted_label ? <CheckCircle2 size={12} /> : null}
            {result.true_label}
          </Badge>
          {result.true_label !== result.predicted_label && (
            <span className="text-xs text-ink-muted">
              (model predicted {result.predicted_label} — a real, reportable miss, not hidden)
            </span>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Reconstruction MSE" value={fmt(result.reconstruction_mse, 5)} />
        <Metric
          label="Prototype distance"
          value={result.new_class_distance != null ? fmt(result.new_class_distance, 4) : '—'}
        />
        <Metric
          label="NEW CLASS threshold"
          value={result.new_class_threshold != null ? fmt(result.new_class_threshold, 4) : '—'}
        />
        <Metric label="Client" value={result.client_id != null ? `#${result.client_id}` : 'shared'} />
      </div>

      <div>
        <p className="label-caps mb-2">Top classes</p>
        <div className="space-y-1.5">
          {result.top_k.map((entry) => (
            <div key={entry.label} className="flex items-center gap-2">
              <span className="w-32 shrink-0 truncate text-xs">{entry.label}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-alt">
                <div
                  className="h-full rounded-full bg-accent"
                  style={{ width: `${Math.max(entry.probability * 100, 1)}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-xs text-ink-muted">
                {fmtPercent(entry.probability)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3 text-xs text-ink-muted">
        <span className="font-mono">{result.model_run}</span>
        <span>·</span>
        <span>{result.mechanism}</span>
        <span>·</span>
        <span>{result.source.replace('_', ' ')}</span>
        <ProvenanceTag provenance={result.provenance} />
        <span className="ml-auto font-mono">{new Date(result.timestamp).toLocaleTimeString()}</span>
      </div>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface-alt/50 p-2.5">
      <p className="text-2xs text-ink-muted">{label}</p>
      <p className="mt-0.5 font-mono text-sm font-medium">{value}</p>
    </div>
  );
}
