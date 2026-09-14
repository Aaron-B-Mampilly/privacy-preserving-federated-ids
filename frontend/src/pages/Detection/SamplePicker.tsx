import { useState } from 'react';
import { FileUp, Shuffle } from 'lucide-react';
import { api } from '@/services/api';
import { useApi } from '@/hooks/useApi';
import { useDataset } from '@/hooks/useDataset';
import { Badge, Button, Card, EmptyState, ErrorState, LoadingState, cx } from '@/components/primitives';
import type { SampleItem, SamplesResponse } from '@/types/api';

type Split = 'test' | 'zero_day_holdout';

/** Prepared samples only — never a manual-entry form. A real input is a
 *  10×70 or 10×115 preprocessed float matrix, and letting someone type
 *  700 numbers would be theatre, not a feature. */
export default function SamplePicker({
  onSelect,
  selected,
}: {
  onSelect: (sample: SampleItem) => void;
  selected: SampleItem | null;
}) {
  const { dataset, scope } = useDataset();
  const [split, setSplit] = useState<Split>('test');
  const [label, setLabel] = useState<string>('');

  const { data, error, loading, refresh } = useApi<SamplesResponse>(
    () => api.samples({ dataset, scope, split, label: label || undefined, limit: 24 }),
    [dataset, scope, split, label],
  );

  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-md border border-border p-0.5">
          {(['test', 'zero_day_holdout'] as Split[]).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => { setSplit(s); setLabel(''); }}
              className={cx(
                'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                split === s ? 'bg-accent-soft text-accent-strong' : 'text-ink-muted hover:text-ink',
              )}
            >
              {s === 'test' ? 'Held-out test' : 'Zero-day holdout'}
            </button>
          ))}
        </div>

        {data && data.labels_available.length > 0 && (
          <select
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="rounded-md border border-border bg-surface px-2 py-1 text-xs"
          >
            <option value="">All classes</option>
            {data.labels_available.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        )}

        {data && (
          <span className="text-2xs text-ink-muted">
            {data.total.toLocaleString()} sequences available
          </span>
        )}

        <Button variant="ghost" icon={<Shuffle size={13} />} onClick={refresh} className="ml-auto">
          Shuffle
        </Button>
      </div>

      {loading && <LoadingState label="Loading samples" />}
      {error && <ErrorState error={error} onRetry={refresh} />}
      {data && data.samples.length === 0 && <EmptyState title="No samples match this filter" />}

      {data && data.samples.length > 0 && (
        <div className="grid max-h-72 grid-cols-2 gap-1.5 overflow-y-auto sm:grid-cols-3">
          {data.samples.map((sample) => {
            const isSelected = selected?.sequence_index === sample.sequence_index;
            return (
              <button
                key={sample.sequence_index}
                type="button"
                onClick={() => onSelect(sample)}
                className={cx(
                  'rounded-md border p-2 text-left text-xs transition-colors',
                  isSelected
                    ? 'border-accent bg-accent-soft'
                    : 'border-border bg-surface hover:bg-surface-alt',
                )}
              >
                <p className="truncate font-medium">{sample.label}</p>
                <p className="mt-0.5 truncate font-mono text-2xs text-ink-muted">
                  #{sample.sequence_index} · client {sample.client_id}
                </p>
              </button>
            );
          })}
        </div>
      )}

      {split === 'zero_day_holdout' && (
        <Badge tone="warning" className="w-fit">
          <FileUp size={11} /> True novel-attack classes — expect NEW CLASS
        </Badge>
      )}
    </Card>
  );
}
