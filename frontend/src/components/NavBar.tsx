/**
 * components/NavBar.tsx
 *
 * Shared top navigation bar used on Overview and Inference pages.
 * Matches the Figma design: logo left, tabs centre, icons+avatar right.
 */

import { NavLink, useNavigate } from 'react-router-dom';
import { logout, getUsername } from '../api/auth';
import './NavBar.css';

export default function NavBar() {
  const navigate = useNavigate();
  const username = getUsername();
  // Show first letter of username (or first letter before @)
  const initials = (username.split('@')[0] ?? username).charAt(0).toUpperCase();

  async function handleLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  return (
    <header className="navbar">
      <div className="navbar-inner">
        {/* ── Logo ── */}
        <span className="navbar-logo">Herbarium AI</span>

        {/* ── Nav tabs ── */}
        <nav className="navbar-tabs" aria-label="Primary navigation">
          <NavLink
            to="/overview"
            id="nav-overview"
            className={({ isActive }) =>
              `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
            }
          >
            Overview
          </NavLink>
          <NavLink
            to="/inference"
            id="nav-inference"
            className={({ isActive }) =>
              `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
            }
          >
            Inference
          </NavLink>
          <NavLink
            to="/history"
            id="nav-history"
            className={({ isActive }) =>
              `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
            }
          >
            History
          </NavLink>
          <NavLink
            to="/library"
            id="nav-library"
            className={({ isActive }) =>
              `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
            }
          >
            Library
          </NavLink>
        </nav>

        {/* ── Right actions ── */}
        <div className="navbar-actions">
          {/* Bell */}
          <button
            className="navbar-icon-btn"
            aria-label="Notifications"
            id="nav-bell"
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
              <path d="M13.73 21a2 2 0 0 1-3.46 0" />
            </svg>
          </button>

          {/* Settings */}
          <button
            className="navbar-icon-btn"
            aria-label="Settings"
            id="nav-settings"
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>

          {/* Logout */}
          <button
            className="navbar-icon-btn"
            aria-label="Logout"
            id="nav-logout"
            onClick={() => void handleLogout()}
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>

          {/* Avatar */}
          <div
            className="navbar-avatar"
            aria-label={`Logged in as ${username}`}
            id="nav-avatar"
            title={username}
          >
            {initials}
          </div>
        </div>
      </div>
    </header>
  );
}
