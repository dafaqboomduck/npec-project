/**
 * api/auth.ts
 *
 * Authentication layer — wraps POST /auth/token, /auth/refresh, /auth/logout.
 * Token storage uses localStorage. All protected requests should attach
 * the Authorization header via authHeader().
 *
 * API base: http://localhost:8000/api/v1  (see VITE_API_URL env var)
 */

import { apiClient } from './client';

// ---------------------------------------------------------------------------
// Types (mirroring api_schema.md AuthResponse)
// ---------------------------------------------------------------------------

export interface AuthTokens {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
  refresh_token: string;
  role: 'user' | 'admin';
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/**
 * POST /auth/token
 * OAuth2 Password flow — returns JWT access + refresh tokens.
 */
export async function login(
  username: string,
  password: string,
): Promise<AuthTokens> {
  const body = new URLSearchParams({
    username,
    password,
    grant_type: 'password',
  });

  return apiClient<AuthTokens>('/auth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  }).catch((err) => {
    // Auth endpoint has specific fallback error text
    if (err.message.includes('API Error')) {
      throw new Error('Invalid credentials. Please try again.');
    }
    throw err;
  });
}

/**
 * POST /auth/refresh
 * Exchange the stored refresh token for a new access + refresh token pair.
 * Old refresh token is immediately invalidated.
 */
export async function refreshToken(): Promise<AuthTokens> {
  const token = localStorage.getItem('refresh_token');
  if (!token) throw new Error('No refresh token stored');

  const tokens = await apiClient<AuthTokens>('/auth/refresh', {
    method: 'POST',
    body: { refresh_token: token },
  }).catch(() => {
    throw new Error('Session expired. Please log in again.');
  });

  saveTokens(tokens, getUsername());
  return tokens;
}

/**
 * POST /auth/logout
 * Revokes the refresh token server-side (required in body).
 * The access token expires naturally (~1 h); we discard it locally immediately.
 * Fire-and-forget network call — the UI clears regardless of server response.
 */
export async function logout(): Promise<void> {
  const refreshToken = localStorage.getItem('refresh_token');

  // Always clear local state first so the UI reacts immediately
  clearTokens();

  if (refreshToken) {
    // Swallow errors — even if the server is unreachable the session is gone locally
    await apiClient<void>('/auth/logout', {
      method: 'POST',
      body: { refresh_token: refreshToken },
    }).catch(() => undefined);
  }
}

// ---------------------------------------------------------------------------
// Token storage helpers
// ---------------------------------------------------------------------------

/**
 * Persist token pair and username to localStorage after successful login.
 * NOTE: The API schema returns `role` but no display name. We store the
 * raw username from the login form and use it for greetings until a /me
 * endpoint is available.
 */
export function saveTokens(tokens: AuthTokens, username: string): void {
  localStorage.setItem('access_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);
  localStorage.setItem('role', tokens.role);
  localStorage.setItem('username', username);
}

export function clearTokens(): void {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('role');
  localStorage.removeItem('username');
}

export function getAccessToken(): string | null {
  return localStorage.getItem('access_token');
}

/** Returns the stored username (email or display name typed at login). */
export function getUsername(): string {
  return localStorage.getItem('username') ?? 'User';
}

export function getRole(): 'user' | 'admin' {
  return (localStorage.getItem('role') as 'user' | 'admin') ?? 'user';
}

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

/**
 * Returns the Authorization header object for use in protected fetch calls.
 * Usage: fetch(url, { headers: authHeader() })
 */
export function authHeader(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
