/* eslint-disable react-refresh/only-export-components */
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
import Dashboard from './pages/Dashboard';
import Inference from './pages/Inference';
import History from './pages/History';
import Library from './pages/Library';
import Models from './pages/Models';
import Lab from './pages/Lab';
import { isAuthenticated, getRole } from './api/auth';
import { InferenceJobsProvider } from './context/InferenceJobsContext';
import { ThemeProvider } from './context/ThemeContext';

/** Redirects unauthenticated users to /login. */
function RequireAuth({ children }: { children: React.ReactNode }) {
  return isAuthenticated() ? <>{children}</> : <Navigate to="/login" replace />;
}

/** Redirects non-admin authenticated users to /overview. */
function RequireAdmin({ children }: { children: React.ReactNode }) {
  return isAuthenticated() && getRole() === 'admin' ? (
    <>{children}</>
  ) : (
    <Navigate to="/overview" replace />
  );
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
          path="/dashboard"
          element={
            <RequireAuth>
              <Dashboard />
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
        <Route
          path="/lab"
          element={
            <RequireAuth>
              <Lab />
            </RequireAuth>
          }
        />
        <Route
          path="/models"
          element={
            <RequireAdmin>
              <Models />
            </RequireAdmin>
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
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </InferenceJobsProvider>
  </StrictMode>,
);
