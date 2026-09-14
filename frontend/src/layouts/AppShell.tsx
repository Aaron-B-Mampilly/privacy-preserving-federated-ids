import { useEffect, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  ExternalLink,
  LayoutDashboard,
  Moon,
  Radar,
  ScanSearch,
  Settings as SettingsIcon,
  Shield,
  Sun,
} from 'lucide-react';
import { useDataset } from '@/hooks/useDataset';
import { cx } from '@/components/primitives';
import { GRAFANA_URL } from '@/services/api';

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/detection', label: 'Intrusion Detection', icon: ScanSearch },
  { to: '/federated', label: 'Federated Learning', icon: Cpu },
  { to: '/privacy', label: 'Privacy & Security', icon: Shield },
  { to: '/drift', label: 'Drift Monitor', icon: Radar },
  { to: '/research', label: 'Research Results', icon: BarChart3 },
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
];

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      const stored = window.localStorage.getItem('fedpda.theme');
      if (stored === 'light' || stored === 'dark') return stored;
    } catch { /* ignore */ }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try {
      window.localStorage.setItem('fedpda.theme', theme);
    } catch { /* ignore */ }
  }, [theme]);

  return { theme, toggle: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')) };
}

function ConnectionPill() {
  const { connection } = useDataset();
  const meta = {
    connected: { label: 'Connected', dot: 'bg-good', text: 'text-good' },
    connecting: { label: 'Connecting', dot: 'bg-finding animate-pulse', text: 'text-finding' },
    disconnected: { label: 'Disconnected', dot: 'bg-danger', text: 'text-danger' },
  }[connection];

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs font-medium">
      <span className={cx('h-1.5 w-1.5 rounded-full', meta.dot)} />
      <span className={meta.text}>{meta.label}</span>
    </span>
  );
}

/** The single place the active dataset is chosen. Scope options are
 *  driven by the dataset, so an impossible pair cannot be selected. */
function DatasetSelector() {
  const { datasets, dataset, scope, setDataset, setScope, datasetInfo } = useDataset();

  return (
    <div className="flex items-center gap-2">
      <Database size={14} className="text-ink-muted" />
      <select
        aria-label="Dataset"
        value={dataset}
        onChange={(e) => setDataset(e.target.value)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-xs font-medium"
      >
        {datasets.map((d) => (
          <option key={d.id} value={d.id}>{d.label}</option>
        ))}
      </select>
      {datasetInfo && datasetInfo.scopes.length > 1 && (
        <select
          aria-label={datasetInfo.scope_label}
          value={scope}
          onChange={(e) => setScope(e.target.value)}
          className="rounded-md border border-border bg-surface px-2 py-1 font-mono text-xs"
        >
          {datasetInfo.scopes.map((s) => (
            <option key={s} value={s}>α = {s}</option>
          ))}
        </select>
      )}
    </div>
  );
}

export default function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const { theme, toggle } = useTheme();
  const { health, datasetInfo, scope, connection } = useDataset();

  return (
    <div className="flex min-h-screen bg-bg text-ink">
      <aside
        className={cx(
          'sticky top-0 flex h-screen shrink-0 flex-col border-r border-border bg-surface transition-[width] duration-200',
          collapsed ? 'w-[68px]' : 'w-[236px]',
        )}
      >
        <div className="flex items-center gap-2.5 border-b border-border px-4 py-4">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-accent text-white">
            <Shield size={17} />
          </span>
          {!collapsed && (
            <div className="min-w-0">
              <p className="truncate font-display text-sm font-semibold leading-tight">FedPDA-IDS</p>
              <p className="truncate text-2xs text-ink-muted">Federated IDS platform</p>
            </div>
          )}
        </div>

        <nav className="flex-1 space-y-0.5 overflow-y-auto p-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={collapsed ? label : undefined}
              className={({ isActive }) =>
                cx(
                  'flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-accent-soft font-medium text-accent-strong'
                    : 'text-ink-muted hover:bg-surface-alt hover:text-ink',
                )
              }
            >
              <Icon size={16} className="shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-border p-2">
          <a
            href={health?.grafana_url ?? GRAFANA_URL}
            target="_blank"
            rel="noreferrer"
            title="Open Grafana"
            className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm text-ink-muted hover:bg-surface-alt hover:text-ink"
          >
            <Activity size={16} className="shrink-0" />
            {!collapsed && (
              <>
                <span className="truncate">Open Monitoring</span>
                <ExternalLink size={12} className="ml-auto shrink-0" />
              </>
            )}
          </a>
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-sm text-ink-muted hover:bg-surface-alt hover:text-ink"
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>

        {!collapsed && (
          <div className="space-y-1 border-t border-border px-3 py-3 text-2xs text-ink-muted">
            <div className="flex items-center justify-between">
              <span>API</span>
              <span className={connection === 'connected' ? 'text-good' : 'text-danger'}>
                {connection === 'connected' ? 'online' : connection}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>Model</span>
              <span className={health?.checkpoints_available ? 'text-good' : 'text-ink-muted'}>
                {health?.checkpoints_available ? 'loaded' : 'unavailable'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>Results</span>
              <span>{health?.results_available ?? '—'} files</span>
            </div>
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-border bg-surface/95 px-5 py-3 backdrop-blur">
          <DatasetSelector />
          {datasetInfo && (
            <span className="hidden text-xs text-ink-muted sm:inline">
              {datasetInfo.label}
              {datasetInfo.scopes.length > 1 ? ` · α=${scope}` : ` · ${scope}`}
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <ConnectionPill />
            <button
              type="button"
              onClick={toggle}
              aria-label="Toggle theme"
              className="rounded-md border border-border bg-surface p-1.5 text-ink-muted hover:text-ink"
            >
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            </button>
          </div>
        </header>

        <main className="flex-1 px-5 py-6 lg:px-8">
          <div className="mx-auto w-full max-w-[1400px] animate-fade-in">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
