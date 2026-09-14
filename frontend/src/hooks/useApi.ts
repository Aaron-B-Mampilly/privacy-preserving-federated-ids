import { useCallback, useEffect, useRef, useState } from 'react';
import { toApiError } from '@/services/api';
import type { ApiError } from '@/types/api';

interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

/**
 * Fetch-on-mount with loading/error/empty states and a manual refresh.
 *
 * `deps` controls refetching -- every caller passes the active dataset
 * and scope, which is what guarantees a scope switch never leaves one
 * dataset's numbers on screen while another dataset's request is in
 * flight (stale responses are discarded by the request counter).
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  options: { enabled?: boolean } = {},
): AsyncState<T> & { refresh: () => void } {
  const { enabled = true } = options;
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: enabled });
  const requestId = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(() => {
    if (!enabled) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const id = ++requestId.current;
    setState((prev) => ({ ...prev, loading: true, error: null }));

    fetcherRef
      .current()
      .then((data) => {
        if (id === requestId.current) setState({ data, error: null, loading: false });
      })
      .catch((err) => {
        if (id === requestId.current) setState({ data: null, error: toApiError(err), loading: false });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    run();
  }, [run]);

  return { ...state, refresh: run };
}

/**
 * Polls only while `active` is true. Used for the retrain job and the
 * backend connection indicator -- the FL and results data are static
 * completed artifacts and are deliberately NOT polled.
 */
export function usePolling(callback: () => void, intervalMs: number, active: boolean) {
  const saved = useRef(callback);
  saved.current = callback;

  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => saved.current(), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs, active]);
}
