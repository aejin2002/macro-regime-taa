"""Environment settings and YAML configuration loading.

FRED_API_KEY is read from the environment (via .env) and never hardcoded.
Everything else that a researcher might want to tweak (series IDs, window
lengths, lags, targets) lives in config/default.yaml so it can be audited
and changed without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class MissingApiKeyError(RuntimeError):
    """Raised when FRED_API_KEY is not configured."""


class Settings(BaseSettings):
    """Process-level settings sourced from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "cache")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    external_dir: Path = Field(default=PROJECT_ROOT / "data" / "external")

    def require_fred_api_key(self) -> str:
        if not self.fred_api_key:
            raise MissingApiKeyError(
                "FRED_API_KEY is missing. Create a .env file from .env.example "
                "and add your FRED API key."
            )
        return self.fred_api_key


def get_settings() -> Settings:
    return Settings()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML pipeline configuration (series IDs, windows, lags, targets)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    return config
