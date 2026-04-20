/**
 * pages/Login.tsx
 *
 * Login page — matches the Figma design exactly.
 * Calls POST /auth/token (OAuth2 password flow) and redirects to /overview.
 *
 * Route: /login
 */

import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { login, saveTokens, clearTokens } from '../api/auth';
import './Login.css';

/** Only used during development when the backend is not yet running. */
const IS_DEV = import.meta.env.DEV;

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!username.trim() || !password.trim()) {
      setError('Please enter your username and password.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const tokens = await login(username.trim(), password);
      saveTokens(tokens, username.trim());
      navigate('/overview', { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  /**
   * Demo mode — bypasses the real API and injects fake tokens into
   * localStorage so you can explore the UI without a running backend.
   * Only available in development (import.meta.env.DEV).
   */
  function handleDemoMode() {
    clearTokens();
    // Fake token — the API will still reject protected requests, but
    // the router auth guard will let you through and all UI states
    // will show their graceful offline fallbacks.
    localStorage.setItem('access_token', 'demo-preview-token');
    localStorage.setItem('refresh_token', 'demo-refresh-token');
    localStorage.setItem('role', 'user');
    localStorage.setItem('username', 'dr.demo@npec.nl');
    navigate('/overview', { replace: true });
  }

  return (
    <div className="login-page">
      <h1 className="login-title">Herbarium AI</h1>

      <div className="login-card">
        <p className="login-card-heading">Login</p>

        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="login-field">
            <label className="login-label" htmlFor="login-username">
              USERNAME
            </label>
            <input
              className="login-input"
              id="login-username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
            />
          </div>

          <div className="login-field">
            <label className="login-label" htmlFor="login-password">
              PASSWORD
            </label>
            <input
              className="login-input"
              id="login-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
          </div>

          {error && (
            <p className="login-error" role="alert">
              {error}
            </p>
          )}

          <button
            className="login-submit"
            id="login-submit-button"
            type="submit"
            disabled={loading}
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        {IS_DEV && (
          <div className="login-demo">
            <span className="login-demo-divider">or</span>
            <button
              type="button"
              className="login-demo-btn"
              id="login-demo-button"
              onClick={handleDemoMode}
            >
              Use Demo Mode
              <span className="login-demo-tag">DEV ONLY</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
