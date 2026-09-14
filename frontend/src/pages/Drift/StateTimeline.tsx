import { CheckCircle2, RefreshCw, Radar, ShieldCheck } from 'lucide-react';
import { cx } from '@/components/primitives';
import type { DriftState } from '@/types/api';

const STAGES: Array<{ id: DriftState; label: string; icon: typeof Radar }> = [
  { id: 'normal', label: 'Normal', icon: ShieldCheck },
  { id: 'drift_detected', label: 'Drift detected', icon: Radar },
  { id: 'retraining', label: 'Retraining', icon: RefreshCw },
  { id: 'recovered', label: 'Recovered', icon: CheckCircle2 },
];

const ORDER: DriftState[] = ['normal', 'drift_detected', 'retraining', 'recovered'];

/** A simple left-to-right state machine: NORMAL -> DRIFT DETECTED ->
 *  RETRAINING -> RECOVERED. `warning` sits between normal and detected
 *  visually but is rare in practice for this project's binary trigger. */
export default function StateTimeline({ state }: { state: DriftState }) {
  const currentIndex = ORDER.indexOf(state === 'warning' ? 'normal' : state);

  return (
    <div className="flex items-stretch">
      {STAGES.map((stage, i) => {
        const isPast = i < currentIndex;
        const isCurrent = i === currentIndex;
        const Icon = stage.icon;
        return (
          <div key={stage.id} className="flex flex-1 items-center">
            <div className="flex flex-col items-center gap-2">
              <div
                className={cx(
                  'grid h-11 w-11 place-items-center rounded-full border-2 transition-colors',
                  isCurrent && 'border-accent bg-accent-soft text-accent-strong animate-fade-in',
                  isPast && 'border-good bg-good-soft text-good',
                  !isCurrent && !isPast && 'border-border bg-surface text-ink-muted',
                )}
              >
                <Icon size={18} className={isCurrent && stage.id === 'retraining' ? 'animate-spin' : ''} />
              </div>
              <span
                className={cx(
                  'text-2xs font-medium uppercase tracking-wide',
                  isCurrent ? 'text-accent-strong' : isPast ? 'text-good' : 'text-ink-muted',
                )}
              >
                {stage.label}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <div
                className={cx(
                  'mx-2 h-0.5 flex-1 rounded-full transition-colors',
                  i < currentIndex ? 'bg-good' : 'bg-border',
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
