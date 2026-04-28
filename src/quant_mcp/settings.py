"""Environment-backed application configuration.

Settings define global runtime boundaries such as artifact paths, MCP HTTP
binding, and live-trading risk limits. Protocol registration and business logic
should read this module, not mutate it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:\\Users\\goku\\Documents\\quant_mcp_repo\\.env", verbose=True)  # Reads .env from project root

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

    kraken_rest_base_url: str = "https://api.kraken.com"
    kraken_ws_public_url: str = "wss://ws.kraken.com/v2"
    kraken_ws_auth_url: str = "wss://ws-auth.kraken.com/v2"

    enable_live_trading: bool = False
    require_explicit_live_mode: bool = True
    allowed_live_symbols: list[str] = Field(default_factory=lambda: ["BTC/USD", "SOL/USD"])
    max_live_allocation_pct: float = 0.02
    max_single_trade_risk_pct: float = 0.01

    kraken_api_key: str = Field(os.getenv("KRAKEN_API_KEY", ""))
    kraken_api_secret: str = Field(os.getenv("KRAKEN_API_SECRET", ""))


def get_settings() -> AppSettings:
    settings = AppSettings()
    # Directory creation is centralized here so services can assume their roots exist.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
