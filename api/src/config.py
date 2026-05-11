"""Configuration for FastAPI."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings."""

    model_config = ConfigDict(env_file=".env", extra="ignore")

    api_version: str = "0.1.0"
    database_url: str
    secret_key: str = "dev-secret-key-replace-in-prod"  # default so .env is optional in dev
    algorithm: str = "HS256"


settings = Settings()
