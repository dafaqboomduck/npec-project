/**
 * api/client.ts
 *
 * Central API client module that reads VITE_API_URL from the environment
 * and handles all backend calls. This ensures consistent networking
 * behavior across local dev, Docker, and Portainer.
 */

export const BASE_URL = import.meta.env.VITE_API_URL ?? '/api/v1';

function getAuthHeader(): Record<string, string> {
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

interface ClientOptions extends Omit<RequestInit, 'body'> {
  body?: BodyInit | null | Record<string, unknown> | unknown[];
}

/**
 * Core fetch wrapper that automatically handles the base URL, authentication
 * headers, and JSON serialization.
 */
export async function apiClient<T>(
  endpoint: string,
  options: ClientOptions = {},
): Promise<T> {
  const { headers, signal, ...customOptions } = options;

  const config: RequestInit = {
    ...customOptions,
    body: options.body as BodyInit | null | undefined,
    signal,
    headers: {
      ...getAuthHeader(),
      ...headers,
    },
  };

  // Only stringify the body if it's a plain object (not FormData or URLSearchParams)
  if (
    config.body &&
    !(config.body instanceof FormData) &&
    !(config.body instanceof URLSearchParams)
  ) {
    config.body = JSON.stringify(config.body);
    config.headers = {
      'Content-Type': 'application/json',
      ...config.headers,
    };
  }

  const response = await fetch(`${BASE_URL}${endpoint}`, config);

  if (!response.ok) {
    // If the token is invalid/expired and this isn't a login attempt itself,
    // clear the session state and redirect directly to the login page.
    if (response.status === 401 && endpoint !== '/auth/token') {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('role');
      localStorage.removeItem('username');
      window.location.href = '/login';

      // Delay throw to prevent UI errors while the browser navigates away
      await new Promise((r) => setTimeout(r, 10000));
    }

    const errorData = (await response.json().catch(() => ({}))) as {
      message?: string;
    };
    throw new Error(errorData.message ?? `API Error (${response.status})`);
  }

  // Handle empty responses
  if (response.status === 204) {
    return {} as T;
  }

  return response.json() as Promise<T>;
}

