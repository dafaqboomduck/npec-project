/**
 * main.tsx
 *
 * Application entry point. Sets up the React router with:
 *   /login        — Login page (public)
 *   /overview     — Home/Overview dashboard (protected)
 *   /inference    — Inference Lab upload page (protected)
 *   /             — Redirects to /overview if authenticated, else /login
 *
 * Routes to add in future sprints:
 *   /results      — Run history browser (pending rewrite against new API)
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import './styles/global.css';

import Login from './pages/Login';
import Overview from './pages/Overview';
import Inference from './pages/Inference';
import History from './pages/History';
import Library from './pages/Library';
import { isAuthenticated } from './api/auth';
import { InferenceJobsProvider } from './context/InferenceJobsContext';

/** Redirects unauthenticated users to /login. */
function RequireAuth({ children }: { children: React.ReactNode }) {
  return isAuthenticated() ? <>{children}</> : <Navigate to="/login" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/login" element={<Login />} />

        {/* Protected */}
        <Route
          path="/overview"
          element={
            <RequireAuth>
              <Overview />
            </RequireAuth>
          }
        />
        <Route
          path="/inference"
          element={
            <RequireAuth>
              <Inference />
            </RequireAuth>
          }
        />
        <Route
          path="/history"
          element={
            <RequireAuth>
              <History />
            </RequireAuth>
          }
        />
        <Route
          path="/library"
          element={
            <RequireAuth>
              <Library />
            </RequireAuth>
          }
        />

        {/* Default redirect */}
        <Route
          path="/"
          element={
            <Navigate to={isAuthenticated() ? '/overview' : '/login'} replace />
          }
        />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element #root not found in index.html');

createRoot(rootElement).render(
  <StrictMode>
    <InferenceJobsProvider>
      <App />
    </InferenceJobsProvider>
  </StrictMode>,
);
