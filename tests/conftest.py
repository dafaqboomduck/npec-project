"""Root pytest configuration.

Ensures environment variables required by `api.src.config.Settings` are set
*before* any test module imports `api.src.*`. Pytest discovers and loads this
conftest before collecting test modules, so the `os.environ` writes here run
at the right time without needing a real `.env` file on the runner.

Also forces the Selector asyncio event loop on Windows. The default Proactor
loop deadlocks `starlette.testclient` + `anyio.BlockingPortal` during FastAPI
lifespan startup on Python 3.12 (Starlette issue #2008), which manifests as
the TestClient hanging forever inside `portal.call(self.wait_startup)`.
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
