"""Dataset ingestion and profiling service.

This service owns Kraken OHLC persistence rules: dataset identity, closed-candle
filtering, append-only refresh, and profiling. It does not build strategy logic.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_mcp.adapters.kraken.public_client import KrakenPublicClient
from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.domain.dataset import (
    DatasetProfile,
    DatasetVersion,
    IngestMarketDataRequest,
    RefreshDatasetRequest,
)
from quant_mcp.settings import AppSettings


class DatasetService:
    def __init__(self, settings: AppSettings, client: KrakenPublicClient | None = None) -> None:
        self.settings = settings
        self.client = client or KrakenPublicClient()
        self.store = ParquetStore(settings.data_dir)

    @staticmethod
    def dataset_id(symbol: str, interval_minutes: int) -> str:
        return f"{symbol.replace('/', '_').lower()}_{interval_minutes}m"

    def dataset_path(self, dataset_id: str) -> Path:
        return self.settings.data_dir / f"datasets/{dataset_id}.parquet"

    async def ingest_market_data(self, request: IngestMarketDataRequest) -> DatasetVersion:
        candles = await self.client.fetch_ohlc(
            symbol=request.symbol,
            interval_minutes=request.interval_minutes,
            since_unix=request.since_unix,
        )
        frame = pd.DataFrame([c.model_dump(mode="json") for c in candles])
        if frame.empty:
            raise ValueError("No candles returned from Kraken")
        # Persist only closed bars so downstream validation never sees mutable exchange candles.
        frame = frame[frame["closed"] == True].copy()  # noqa: E712
        frame = frame.sort_values("ts_open").drop_duplicates(
            subset=["symbol", "interval_minutes", "ts_open"]
        )
        dataset_id = self.dataset_id(request.symbol, request.interval_minutes)
        path = self.store.write_frame(f"datasets/{dataset_id}.parquet", frame)
        return DatasetVersion(
            dataset_id=dataset_id,
            symbol=request.symbol,
            interval_minutes=request.interval_minutes,
            version=str(uuid4()),
            row_count=len(frame),
            path=str(path),
            notes="Initial ingest from Kraken public OHLC",
        )

    async def refresh_dataset(self, request: RefreshDatasetRequest) -> DatasetVersion:
        dataset_id = self.dataset_id(request.symbol, request.interval_minutes)
        path = self.dataset_path(dataset_id)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()

        since_unix = None
        if not existing.empty:
            last_open = pd.to_datetime(existing["ts_open"].max(), utc=True)
            # Re-fetch from the last known open time to overlap one bar, then dedupe below.
            since_unix = int(last_open.timestamp())

        new_candles = await self.client.fetch_ohlc(request.symbol, request.interval_minutes, since_unix)
        new_frame = pd.DataFrame([c.model_dump(mode="json") for c in new_candles])
        if not new_frame.empty:
            # The same closed-candle rule applies on refresh to preserve append-only semantics.
            new_frame = new_frame[new_frame["closed"] == True].copy()  # noqa: E712

        combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
        if combined.empty:
            raise ValueError("Refresh produced no rows")

        # Sorting plus dedupe makes refresh idempotent when Kraken returns an overlapping candle.
        combined = combined.sort_values("ts_open").drop_duplicates(
            subset=["symbol", "interval_minutes", "ts_open"], keep="first"
        )
        out_path = self.store.write_frame(f"datasets/{dataset_id}.parquet", combined)
        return DatasetVersion(
            dataset_id=dataset_id,
            symbol=request.symbol,
            interval_minutes=request.interval_minutes,
            version=str(uuid4()),
            row_count=len(combined),
            path=str(out_path),
            notes="Append-only refresh using closed candles only",
        )

    def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        frame = pd.read_parquet(self.dataset_path(dataset_id))
        duplicate_rows = int(frame.duplicated(subset=["symbol", "interval_minutes", "ts_open"]).sum())
        null_counts = {column: int(value) for column, value in frame.isna().sum().items()}
        return DatasetProfile(
            dataset_id=dataset_id,
            symbol=str(frame["symbol"].iloc[0]),
            interval_minutes=int(frame["interval_minutes"].iloc[0]),
            rows=len(frame),
            first_ts=pd.to_datetime(frame["ts_open"].min(), utc=True).to_pydatetime(),
            last_ts=pd.to_datetime(frame["ts_open"].max(), utc=True).to_pydatetime(),
            duplicate_rows=duplicate_rows,
            null_counts=null_counts,
        )

    def list_dataset_versions(self) -> list[DatasetVersion]:
        versions: list[DatasetVersion] = []
        dataset_dir = self.settings.data_dir / "datasets"
        if not dataset_dir.exists():
            return versions
        for path in dataset_dir.glob("*.parquet"):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            versions.append(
                DatasetVersion(
                    dataset_id=path.stem,
                    symbol=str(frame["symbol"].iloc[0]),
                    interval_minutes=int(frame["interval_minutes"].iloc[0]),
                    version="current",
                    row_count=len(frame),
                    path=str(path),
                )
            )
        return versions
