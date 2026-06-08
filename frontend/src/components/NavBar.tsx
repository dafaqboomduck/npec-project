/**
 * components/NavBar.tsx
 *
 * Shared top navigation bar used on Overview and Inference pages.
 * Matches the Figma design: logo left, tabs centre, icons+avatar right.
 */

import { NavLink, useNavigate, Link } from 'react-router-dom';
import { logout, getUsername, getRole } from '../api/auth';
import SettingsPanel from './SettingsPanel';
import './NavBar.css';
import { useState } from 'react';
import { useInferenceJobs } from '../context/InferenceJobsContext';

export default function NavBar() {
  const navigate = useNavigate();
  const username = getUsername();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const { queueStatus } = useInferenceJobs();
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
        <Link to="/overview" className="navbar-logo" id="nav-logo">
          Herbarium AI
        </Link>

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
            to="/dashboard"
            id="nav-dashboard"
            className={({ isActive }) =>
              `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
            }
          >
            Dashboard
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
            to="/lab"
            className={({ isActive }) =>
              `navbar-link${isActive ? ' navbar-link--active' : ''}`
            }
            id="nav-lab"
          >
            Lab
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
          {getRole() === 'admin' && (
            <NavLink
              to="/models"
              id="nav-models"
              className={({ isActive }) =>
                `navbar-tab${isActive ? ' navbar-tab--active' : ''}`
              }
            >
              Models
            </NavLink>
          )}
        </nav>

        {/* ── Right actions ── */}
        <div className="navbar-actions">
          {/* Inference Queue Status */}
          {(queueStatus.running > 0 || queueStatus.queued > 0) && (
            <div
              className="navbar-queue-status"
              title={`${queueStatus.running} running, ${queueStatus.queued} queued`}
            >
              <span className="navbar-queue-dot" />
              <span className="navbar-queue-text">
                {queueStatus.running} running
                {queueStatus.queued > 0 ? `, ${queueStatus.queued} queued` : ''}
              </span>
            </div>
          )}

          {/* Settings */}
          <button
            className="navbar-icon-btn"
            aria-label="Settings"
            id="nav-settings"
            onClick={() => setIsSettingsOpen(true)}
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

      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </header>
  );
}
