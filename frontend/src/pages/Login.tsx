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
import { login, saveTokens } from '../api/auth';
import './Login.css';

export default function Login() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    if (!username.trim()) {
      setError('Please enter your username.');
      return;
    }

    if (!password.trim()) {
      setError('Please enter your password.');
      document.getElementById('login-password')?.focus();
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
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !password.trim()) {
                  e.preventDefault();
                  document.getElementById('login-password')?.focus();
                }
              }}
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
      </div>
    </div>
  );
}
