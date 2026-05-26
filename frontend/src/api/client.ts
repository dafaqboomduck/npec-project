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
  body?: any;
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
