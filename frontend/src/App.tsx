import { Suspense, lazy } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import AppShell from '@/layouts/AppShell';
import { DatasetProvider } from '@/hooks/useDataset';
import { LoadingState } from '@/components/primitives';
import Dashboard from '@/pages/Dashboard';

// Heavier screens are split out so the first paint only carries the
// dashboard, which is what a demo opens on.
const Research = lazy(() => import('@/pages/Research'));
const Detection = lazy(() => import('@/pages/Detection'));
const FederatedLearning = lazy(() => import('@/pages/FederatedLearning'));
const Privacy = lazy(() => import('@/pages/Privacy'));
const Drift = lazy(() => import('@/pages/Drift'));
const Settings = lazy(() => import('@/pages/Settings'));

export default function App() {
  return (
    <DatasetProvider>
      <BrowserRouter>
        <Suspense fallback={<LoadingState />}>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Dashboard />} />
              <Route path="detection" element={<Detection />} />
              <Route path="federated" element={<FederatedLearning />} />
              <Route path="privacy" element={<Privacy />} />
              <Route path="drift" element={<Drift />} />
              <Route path="research" element={<Research />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </DatasetProvider>
  );
}
