import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { api } from '@/services/api';
import { useApi, usePolling } from '@/hooks/useApi';
import type { DatasetInfo, HealthResponse } from '@/types/api';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected';

interface DatasetContextValue {
  datasets: DatasetInfo[];
  dataset: string;
  scope: string;
  datasetInfo: DatasetInfo | null;
  setDataset: (dataset: string) => void;
  setScope: (scope: string) => void;
  connection: ConnectionState;
  health: HealthResponse | null;
  refreshHealth: () => void;
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

const STORAGE_KEY = 'fedpda.scope';

/**
 * Holds the ONE active (dataset, scope) pair for the whole app.
 *
 * CICIDS2017 and N-BaIoT are two independent federations and are never
 * merged or aggregated together, so there is deliberately no "all
 * datasets" option here -- every screen reads this context and labels
 * which dataset it is showing.
 */
export function DatasetProvider({ children }: { children: ReactNode }) {
  const [dataset, setDatasetState] = useState('cicids2017');
  const [scope, setScopeState] = useState('5');

  const { data: datasets, error: datasetsError } = useApi<DatasetInfo[]>(() => api.datasets(), []);
  const { data: health, error: healthError, refresh: refreshHealth } = useApi<HealthResponse>(
    () => api.health(),
    [],
  );

  // Poll health only -- it is the connection indicator. Everything else
  // on these screens is a completed artifact that cannot change under us.
  usePolling(refreshHealth, 15_000, true);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as { dataset?: string; scope?: string };
        if (parsed.dataset) setDatasetState(parsed.dataset);
        if (parsed.scope) setScopeState(parsed.scope);
      }
    } catch {
      /* private mode / blocked storage -- defaults are fine */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ dataset, scope }));
    } catch {
      /* non-fatal */
    }
  }, [dataset, scope]);

  const setDataset = useCallback(
    (next: string) => {
      setDatasetState(next);
      // Switching dataset must also move to a scope that exists for it,
      // otherwise the next request would 404 on an impossible pair.
      const info = datasets?.find((d) => d.id === next);
      setScopeState(info?.scopes[0] ?? 'main_45');
    },
    [datasets],
  );

  const connection: ConnectionState = healthError
    ? 'disconnected'
    : health
      ? 'connected'
      : 'connecting';

  const value = useMemo<DatasetContextValue>(
    () => ({
      datasets: datasets ?? [],
      dataset,
      scope,
      datasetInfo: datasets?.find((d) => d.id === dataset) ?? null,
      setDataset,
      setScope: setScopeState,
      connection,
      health: health ?? null,
      refreshHealth,
    }),
    [datasets, dataset, scope, setDataset, connection, health, refreshHealth],
  );

  void datasetsError;

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset() {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error('useDataset must be used inside <DatasetProvider>');
  return ctx;
}
