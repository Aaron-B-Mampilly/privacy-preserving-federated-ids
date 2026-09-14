import { useEffect, useMemo, useState } from 'react';
import { BookOpen, Info } from 'lucide-react';
import { api } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import {
  Badge,
  Callout,
  Card,
  ErrorState,
  LoadingState,
  Tabs,
  cx,
} from '@/components/primitives';
import type { ExperimentDetail, ExperimentSummary } from '@/types/api';
import { E1Table, E2Table, E3Table, E4Table, E5Table, E6Table, T7Table, prettyLabel } from './renderers';

type ExperimentId = 'E1' | 'E2' | 'E3' | 'E4' | 'E5' | 'E6' | 'T7';

/** Experiments whose data is keyed by scope and therefore need a scope
 *  picker; E2 is an alpha sweep that already shows every alpha at once. */
const SCOPED: Record<ExperimentId, boolean> = {
  E1: true, E2: false, E3: true, E4: true, E5: true, E6: true, T7: true,
};

export default function Research() {
  const [active, setActive] = useState<ExperimentId>('E1');
  const [scopeKey, setScopeKey] = useState<string>('');

  const { data: list } = useApi<ExperimentSummary[]>(() => api.experiments(), []);
  const { data: detail, error, loading, refresh } = useApi<ExperimentDetail>(
    () => api.experiment(active),
    [active],
  );

  useEffect(() => {
    if (detail?.scopes_covered?.length) {
      setScopeKey((current) =>
        detail.scopes_covered.includes(current) ? current : detail.scopes_covered[0],
      );
    }
  }, [detail]);

  const tabs = useMemo(
    () =>
      (list ?? []).map((e) => ({
        id: e.id as ExperimentId,
        label: e.title,
        disabled: !e.available,
      })),
    [list],
  );

  const body = () => {
    if (!detail || !scopeKey) return null;
    switch (detail.id as ExperimentId) {
      case 'E1': return <E1Table data={detail.data} scopeKey={scopeKey} />;
      case 'E2': return <E2Table data={detail.data} />;
      case 'E3': return <E3Table data={detail.data} scopeKey={scopeKey} />;
      case 'E4': return <E4Table data={detail.data} scopeKey={scopeKey} />;
      case 'E5': return <E5Table data={detail.data} scopeKey={scopeKey} />;
      case 'E6': return <E6Table data={detail.data} scopeKey={scopeKey} />;
      case 'T7': return <T7Table data={detail.data} scopeKey={scopeKey} />;
      default: return null;
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="font-display text-2xl">Research Results</h1>
        <p className="mt-1 text-sm text-ink-muted">
          E1–E6 and T7, read directly from the consolidated results file the thesis tables are
          generated from. Numbers here are never rounded, reshaped or adjusted for display.
        </p>
      </header>

      {tabs.length > 0 && <Tabs tabs={tabs} active={active} onChange={setActive} />}

      {loading && !detail && <LoadingState label="Loading experiment" />}
      {error && <ErrorState error={error} onRetry={refresh} />}

      {detail && (
        <div className="space-y-5">
          <Card>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="accent">{detail.id}</Badge>
                  <h2 className="font-display text-lg">{detail.title}</h2>
                  <span className="text-xs text-ink-muted">{detail.subtitle}</span>
                </div>
                <p className="mt-2 max-w-3xl text-sm text-ink-muted">{detail.description}</p>
              </div>

              {SCOPED[detail.id as ExperimentId] && detail.scopes_covered.length > 1 && (
                <div className="flex flex-wrap gap-1">
                  {detail.scopes_covered.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setScopeKey(s)}
                      className={cx(
                        'rounded-md border px-2.5 py-1 text-xs font-medium transition-colors',
                        scopeKey === s
                          ? 'border-accent bg-accent-soft text-accent-strong'
                          : 'border-border bg-surface text-ink-muted hover:bg-surface-alt',
                      )}
                    >
                      {prettyLabel(s)}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {SCOPED[detail.id as ExperimentId] && detail.scopes_covered.length === 1 && (
              <p className="mt-3 text-xs text-ink-muted">
                Scope: <span className="font-mono">{prettyLabel(detail.scopes_covered[0])}</span>
              </p>
            )}
          </Card>

          <div>{body()}</div>

          <Card>
            <div className="mb-3 flex items-center gap-2">
              <BookOpen size={15} className="text-accent" />
              <h3 className="text-base">Interpretation</h3>
            </div>
            <ul className="space-y-2.5">
              {detail.interpretation.map((point, i) => (
                <li key={i} className="flex gap-2.5 text-sm leading-relaxed">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </Card>

          <Callout kind="mechanism" label="Provenance">
            <p className="flex gap-2">
              <Info size={14} className="mt-0.5 shrink-0" />
              <span>{detail.provenance_note}</span>
            </p>
          </Callout>
        </div>
      )}
    </div>
  );
}
