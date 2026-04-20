## Authentication & Authorisation Scheme
**Cersion:** 0.1
**Author:** Erik Hrina (230395)  
**Refer to:** `docs/api_endpoints.md` - `docs/api_spec_draft.md`

---

### 1. Scheme Overview

| Property | Value |
|----------|-------|
| Scheme | JWT Bearer token |
| Grant type | OAuth2 Password flow |
| Token transport | `Authorization: Bearer <token>` header on every protected request |
| Token format | Signed JWT |
| Roles | `public`, `user`, `admin` |
| Refresh strategy | Rotating refresh tokens - see 2.3 |

All protected endpoints return `401 UNAUTHORIZED` if the token is missing or invalid, and `403 FORBIDDEN` if the token is valid but the role is insufficient. See `docs/api_spec_draft.md` 7.2 for error response shapes.

---

### 2. Token Specification

#### 2.1 Access Token

| Property | Value |
|----------|-------|
| Type | JWT, signed with HS256 |
| Expiry | 3600 seconds (1 hour) |
| Transport | `Authorization: Bearer <token>` header |

**JWT payload:**

| Claimn | Type | Notes |
|--------|------|-------|
| `sub` | `string` | User ID |
| `email` | `string` | User's email |
| `role` | `string` | "user" or "admin" |
| `exp` | `integer` | Unix timestamp of expiry |
| `iat` | `integer` | Unix timestamp of issue |

**Example decoded payload:**
```json
{
  "sub": "usr_a1b2c3d4",
  "email": "dr.hrina@npec.nl",
  "role": "admin",
  "exp": 1745919825,
  "iat": 1745916225
}
```

---

#### 2.2 Refresh Token

| Property | Value |
|----------|-------|
| Type | Opaque token (not a JWT) |
| Expiry | 1 day |
| Storage | Server-side in the database, linked to the user |
| Rotation | Rotated on every use - old token invalidated immediately |
| Transport | Sent in the response body of `POST /auth/token` and `POST /auth/refresh`; stored client-side (see 7) |

The refresh token carries no claims and cannot be decoded. Its only purpose is to get a new access token without re-entering credentials.

---

#### 2.3 Refresh Strategy

When the access token expires the client must call `POST /auth/refresh` with the current refresh token. The server returns a new access token and a new refresh token, and immediately invalidates the old refresh token.

If the refresh token itself has expired (after 24 hours), or has been invalidated (e.g. after logout), `POST /auth/refresh` returns `401 UNAUTHORIZED`. The UI must redirect the user to the login screen.

Logout (`POST /auth/logout`) invalidates both the access token and the refresh token for the current session.

---

### 3. Client Authentication Flow

The sequence every frontend client must implement:

```
1. User submits login form (email + password)
        │
        ▼
2. POST /auth/token
   → Receive access_token (1h) + refresh_token (1d) + role
   → Store access_token in memory (not localStorage)
   → Store refresh_token in an httpOnly cookie or secure storage
        │
        ▼
3. Attach token to every protected request
   Authorization: Bearer <access_token>
        │
        ▼
4. Request succeeds → continue normally
        │
        ├── 401 UNAUTHORIZED received
        │        │
        │        ▼
        │   5. POST /auth/refresh with refresh_token
        │        │
        │        ├── 200 OK → store new access_token and refresh_token
        │        │            retry the original request once
        │        │
        │        └── 401 UNAUTHORIZED → refresh token expired or invalid
        │                               clear all tokens
        │                               redirect user to login screen
        │
        └── 403 FORBIDDEN received
                 │
                 ▼
            User lacks required role - do not retry
            Show "Access denied" message in the UI
            Do not redirect to login (the user IS authenticated)
        │
        ▼
6. User clicks logout
   POST /auth/logout
   → Clear access_token from memory
   → Clear refresh_token from storage
   → Redirect to login screen
```

**Important implementation notes:**
- The access token must be stored in memory (a React state), not in `localStorage` or `sessionStorage`, to reduce XSS exposure.
- The refresh token should be stored in an `httpOnly` cookie since the frontend and backend share a domain.
- The token refresh (step 5) should be transparent to the user - retry the original request automatically after a successful refresh.
- Only retry once after a refresh. If the retried request also returns `401`, treat it as an invalid session and force re-login.

---

### 4. Endpoint Access Table

Every endpoint in `docs/api_endpoints` is explicitly marked with its access level. **Public** endpoints require no token. **User** endpoints require a valid JWT with any role. **Admin** endpoints require a valid JWT with `role: "admin"`.

---

### 5. Role Definitions and Restrictions

#### `public`
No token required. Limited to endpoints that must be accessible without authentication (login, token refresh, health check). Any other access is denied with `401 UNAUTHORIZED`.

#### `user`
A valid JWT with `role: "user"` or `role: "admin"`. Can access all user-level endpoints. Cannot access admin-only endpoints - denied with `403 FORBIDDEN`. For resources with ownership, users may only access their own records - server enforces this by comparing the `sub` claim in the JWT against the record's `owner_id`.

#### `admin`
A valid JWT with `role: "admin"`. Can access all endpoints including admin-only ones. Admin accounts are provisioned manually - there is no self-registration flow for admin accounts.

---

### 6. Rate Limits

Rate limits are enforced per IP for public endpoints and per user (`sub` claim) for authenticated endpoints.

| Role | Limit | Window | Notes |
|------|-------|--------|-------|
| Public (unauthenticated) | 10 requests | per minute | Applies to `/auth/token`, `/auth/refresh`, `/health` - prevents brute-force login attempts |
| User | 120 requests | per minute | General API usage |
| User - `POST /predict` | 20 requests | per minute | Inference is compute-intensive; separate limit to protect the model server |
| Admin | 300 requests | per minute | Higher limit to accommodate monitoring and pipeline tooling |

Requests that exceed the rate limit receive `429 Too Many Requests` with a `Retry-After` header indicating when the limit resets.

---

### 7. Security Notes

- **Token storage:** Access tokens must be stored in memory only - never in `localStorage` or `sessionStorage`. Refresh tokens should be stored in `httpOnly` cookies where possible.
- **Token rotation:** Refresh tokens are rotated on every use. If a refresh token is presented twice (e.g. due to a race condition or theft), the second use returns `401` and the session should be considered compromised.
- **HTTPS only:** All endpoints must be served over HTTPS in on-premise and cloud deployments. HTTP is acceptable for local development only.
- **Token expiry on logout:** `POST /auth/logout` must invalidate the server-side refresh token record immediately. The access token cannot be revoked (it is stateless) but expires within 1 hour.
- **Ownership enforcement:** For predictions and annotations, the server must verify that the `sub` claim in the JWT matches the `owner_id` of the resource before returning or modifying it. This is not enforced by the role system alone.

---

### 8. Open Questions

- [ ] **Rate limit values:** The numbers in 6 are provisional. Confirm with the team before Sprint 2.
- [ ] **Admin provisioning:** No endpoint exists for creating admin accounts. Confirm whether admin accounts are assigned in the database directly, or whether a `POST /admin/users` endpoint is needed.