import { useTheme } from '../context/ThemeContext';
import './SettingsPanel.css';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const { theme, toggleTheme } = useTheme();

  if (!isOpen) return null;

  return (
    <>
      <div className="settings-overlay" onClick={onClose} />
      <aside className="settings-panel">
        <div className="settings-header">
          <h2>Settings</h2>
          <button
            className="settings-close"
            onClick={onClose}
            aria-label="Close settings"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="settings-content">
          <div className="settings-section">
            <h3>Appearance</h3>
            <div className="settings-row">
              <span>Dark Mode</span>
              <button
                className={`theme-toggle ${theme === 'dark' ? 'active' : ''}`}
                onClick={toggleTheme}
                aria-label="Toggle dark mode"
              >
                <div className="theme-toggle-thumb" />
              </button>
            </div>
          </div>

          <div className="settings-section">
            <h3>Account</h3>
            <p className="settings-info">
              Manage your profile and preferences.
            </p>
          </div>
        </div>

        <div className="settings-footer">
          <p>Herbarium AI v0.1.0</p>
        </div>
      </aside>
    </>
  );
}
