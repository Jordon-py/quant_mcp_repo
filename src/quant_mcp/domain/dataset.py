"""Dataset and feature-table domain contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from quant_mcp.domain.common import utc_now
from quant_mcp.enums import DatasetStatus


class Candle(BaseModel):
    symbol: str
    interval_minutes: int
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    venue: str = "kraken"
    closed: bool = True


class IngestMarketDataRequest(BaseModel):
    symbol: str
    interval_minutes: int = 60
    since_unix: int | None = None
    max_rows: int = 500


class RefreshDatasetRequest(BaseModel):
    symbol: str
    interval_minutes: int = 60
    max_rows: int = 0


class DatasetProfile(BaseModel):
    dataset_id: str
    symbol: str
    interval_minutes: int
    rows: int
    first_ts: datetime | None
    last_ts: datetime | None
    duplicate_rows: int = 0
    null_counts: dict[str, int] = Field(default_factory=dict)


class DatasetVersion(BaseModel):
    dataset_id: str
    symbol: str
    interval_minutes: int
    version: str
    status: DatasetStatus = DatasetStatus.ACTIVE
    row_count: int
    created_at: datetime = Field(default_factory=utc_now)
    path: str
    notes: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class FeatureTableRequest(BaseModel):
    dataset_id: str
    lookback_fast: int = 10
    lookback_slow: int = 30


class FeatureTableResult(BaseModel):
    dataset_id: str
    feature_path: str
    rows: int
    columns: list[str]
