from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "quant-research-mcp"
    environment: str = "dev"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    log_level: str = "INFO"

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    mcp_path: str = "/mcp"

    enable_live_trading: bool = False
    require_explicit_live_mode: bool = True
    allowed_live_symbols: list[str] = Field(default_factory=lambda: ["BTC/USD", "SOL/USD"])
    max_live_allocation_pct: float = 0.02
    max_single_trade_risk_pct: float = 0.01

    kraken_api_key: str | None = None
    kraken_api_secret: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
