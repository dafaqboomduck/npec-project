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

export interface ClientOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  staleTime?: number;
  bypassCache?: boolean;
}

interface CacheEntry {
  data: unknown;
  timestamp: number;
  staleTime: number;
}

const apiCache = new Map<string, CacheEntry>();

// Token refresh state and queuing variables
let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];
let refreshTimeoutId: number | null = null;

function subscribeTokenRefresh(cb: (token: string) => void) {
  refreshSubscribers.push(cb);
}

function onRefreshed(token: string) {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
}

function getJwtExpiry(token: string): number | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const base64Url = parts[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      window
        .atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join(''),
    );
    const payload = JSON.parse(jsonPayload) as { exp?: number };
    return payload.exp ? payload.exp * 1000 : null; // Convert to milliseconds
  } catch {
    return null;
  }
}

export function scheduleProactiveRefresh(): void {
  if (refreshTimeoutId) {
    clearTimeout(refreshTimeoutId as number);
    refreshTimeoutId = null;
  }

  const accessToken = localStorage.getItem('access_token');
  const refreshToken = localStorage.getItem('refresh_token');
  if (!accessToken || !refreshToken) return;

  const expiry = getJwtExpiry(accessToken);
  if (!expiry) return;

  const now = Date.now();
  const timeUntilExpiry = expiry - now;

  // Refresh 2 seconds before expiry (or 10% of expiry time for very short lifetimes)
  const refreshLeadTime =
    timeUntilExpiry > 20000 ? 10000 : Math.max(1000, timeUntilExpiry / 2);
  const delay = timeUntilExpiry - refreshLeadTime;

  if (delay <= 0) {
    // If the token is already expired or very close, refresh immediately
    void performTokenRefresh()
      .then(() => scheduleProactiveRefresh())
      .catch(() => {});
  } else {
    refreshTimeoutId = window.setTimeout(() => {
      void performTokenRefresh()
        .then(() => scheduleProactiveRefresh())
        .catch(() => {});
    }, delay);
  }
}

async function performTokenRefresh(): Promise<string> {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) {
    throw new Error('No refresh token stored');
  }

  const res = await fetch(`${BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (!res.ok) {
    throw new Error('Failed to refresh session');
  }

  const data = (await res.json()) as {
    access_token: string;
    refresh_token: string;
    role: string;
  };

  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);
  localStorage.setItem('role', data.role);

  // Re-schedule the next proactive refresh
  scheduleProactiveRefresh();

  return data.access_token;
}

/**
 * Core fetch wrapper that automatically handles the base URL, authentication
 * headers, and JSON serialization. Includes an optional in-memory caching system
 * that invalidates automatically on write operations.
 */
export async function apiClient<T>(
  endpoint: string,
  options: ClientOptions = {},
): Promise<T> {
  const { headers, signal, staleTime, bypassCache, ...customOptions } = options;

  const method = (customOptions.method ?? 'GET').toUpperCase();
  const cacheKey = `${method}:${endpoint}`;

  // 1. Read-Cache check
  if (method === 'GET' && staleTime && staleTime > 0 && !bypassCache) {
    const cached = apiCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < cached.staleTime) {
      return cached.data as T;
    }
  }

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
    // If the token is invalid/expired and this isn't an authentication attempt itself,
    // attempt to refresh the access token silently.
    if (
      response.status === 401 &&
      endpoint !== '/auth/token' &&
      endpoint !== '/auth/refresh'
    ) {
      if (isRefreshing) {
        // If a refresh is already in progress, queue this request until it finishes
        try {
          const newToken = await new Promise<string>((resolve, reject) => {
            subscribeTokenRefresh((token) => resolve(token));
            // Safety timeout in case the refresh request hangs
            setTimeout(() => reject(new Error('Refresh timeout')), 10000);
          });

          // Retry the request with the new access token
          const retryConfig: RequestInit = {
            ...config,
            headers: {
              ...config.headers,
              Authorization: `Bearer ${newToken}`,
            },
          };
          const retryResponse = await fetch(
            `${BASE_URL}${endpoint}`,
            retryConfig,
          );
          if (retryResponse.ok) {
            if (retryResponse.status === 204) {
              if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
                apiCache.clear();
              }
              return {} as T;
            }
            return (await retryResponse.json()) as T;
          }
        } catch {
          // If retry fails, continue to default logout logic
        }
      } else {
        isRefreshing = true;
        try {
          const newToken = await performTokenRefresh();
          isRefreshing = false;
          onRefreshed(newToken);

          // Retry the request with the new access token
          const retryConfig: RequestInit = {
            ...config,
            headers: {
              ...config.headers,
              Authorization: `Bearer ${newToken}`,
            },
          };
          const retryResponse = await fetch(
            `${BASE_URL}${endpoint}`,
            retryConfig,
          );
          if (retryResponse.ok) {
            if (retryResponse.status === 204) {
              if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
                apiCache.clear();
              }
              return {} as T;
            }
            return (await retryResponse.json()) as T;
          }
        } catch {
          isRefreshing = false;
          // Refresh failed - clear the session and redirect to /login
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('role');
          localStorage.removeItem('username');
          window.location.href = '/login';
          await new Promise((r) => setTimeout(r, 10000));
          throw new Error('Session expired. Please log in again.');
        }
      }
    }

    // Default logout for auth endpoint failures (or if silent refresh retry still returned 401)
    if (
      response.status === 401 &&
      (endpoint === '/auth/token' || endpoint === '/auth/refresh')
    ) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('role');
      localStorage.removeItem('username');
      window.location.href = '/login';
      await new Promise((r) => setTimeout(r, 10000));
    }

    const errorData = (await response.json().catch(() => ({}))) as {
      message?: string;
    };
    throw new Error(errorData.message ?? `API Error (${response.status})`);
  }

  // Handle empty responses
  if (response.status === 204) {
    // 2. Clear cache on mutations (write operations)
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
      apiCache.clear();
    }
    return {} as T;
  }

  const result = (await response.json()) as T;

  // 3. Write-Cache update
  if (method === 'GET' && staleTime && staleTime > 0) {
    apiCache.set(cacheKey, {
      data: result,
      timestamp: Date.now(),
      staleTime,
    });
  }

  // 4. Clear cache on mutations (write operations)
  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    apiCache.clear();
  }

  return result;
}

/**
 * Resolves artifact URLs (e.g. relative backend paths) to fully qualified URLs.
 * If the URL is already absolute (like a MinIO presigned URL), it is returned unmodified.
 */
export function getArtifactUrl(url: string | undefined | null): string {
  if (!url) return '';
  if (url.startsWith('data:')) {
    return url;
  }

  // If this is already a public, reachable absolute URL, return it unmodified
  if (
    (url.startsWith('http://') || url.startsWith('https://')) &&
    !url.includes('://minio:9000/')
  ) {
    return url;
  }

  let targetUrl = url;

  // If this is an absolute URL pointing to the unreachable MinIO container endpoint,
  // extract the object key and route it through our backend's secure `/files` proxy endpoint.
  if (url.includes('://minio:9000/')) {
    try {
      const parsedUrl = new URL(url);
      const pathname = parsedUrl.pathname;
      const bucketPrefix = '/npec-artifacts/';
      if (pathname.startsWith(bucketPrefix)) {
        targetUrl = pathname.substring(bucketPrefix.length);
      } else {
        const parts = pathname.split('/');
        targetUrl = parts.slice(2).join('/');
      }
    } catch {
      // Fallback in case of parsing error
    }
  }

  // If the target URL is still absolute (e.g., custom external URLs), return it as-is
  if (targetUrl.startsWith('http://') || targetUrl.startsWith('https://')) {
    return targetUrl;
  }

  // Normalize leading slash for relative paths
  const cleanUrl = targetUrl.startsWith('/') ? targetUrl : `/${targetUrl}`;

  // Route it through the /files proxy path under the active API prefix (e.g. /api/v1)
  let apiPath = '';
  if (BASE_URL.startsWith('http')) {
    apiPath = new URL(BASE_URL).pathname;
  } else {
    apiPath = BASE_URL;
  }

  // Strip trailing slash
  if (apiPath.endsWith('/')) {
    apiPath = apiPath.slice(0, -1);
  }

  const fullPath = `${apiPath}/files${cleanUrl}`;

  if (BASE_URL.startsWith('http')) {
    const origin = new URL(BASE_URL).origin;
    return `${origin}${fullPath}`;
  }

  return `${window.location.origin}${fullPath}`;
}
