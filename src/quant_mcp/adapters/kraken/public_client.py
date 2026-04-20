"""Kraken public market-data adapter.

This client is intentionally read-only and unsigned. It converts Kraken OHLC
rows into internal Candle contracts for dataset services.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import httpx

from quant_mcp.domain.dataset import Candle


class KrakenPublicClient:
    """Public market-data adapter. Safe because it never signs or places orders."""

    base_url = "https://api.kraken.com/0/public"

    async def fetch_ohlc(
        self,
        symbol: str,
        interval_minutes: int = 60,
        since_unix: int | None = None,
    ) -> list[Candle]:
        params = {"pair": symbol, "interval": interval_minutes}
        if since_unix is not None:
            params["since"] = since_unix

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{self.base_url}/OHLC", params=params)
            response.raise_for_status()
            payload = response.json()

        if payload.get("error"):
            raise ValueError(f"Kraken OHLC error: {payload['error']}")

        result = payload.get("result", {})
        # Kraken nests candles under the resolved pair key and reserves "last" for pagination.
        pair_key = next((k for k in result.keys() if k != "last"), None)
        if pair_key is None:
            return []

        return [self._to_candle(symbol, interval_minutes, row) for row in result[pair_key]]

    @staticmethod
    def _to_candle(symbol: str, interval_minutes: int, row: Sequence[str | int | float]) -> Candle:
        ts_open = datetime.fromtimestamp(int(float(row[0])), tz=timezone.utc)
        ts_close = datetime.fromtimestamp(int(float(row[0])) + interval_minutes * 60, tz=timezone.utc)
        return Candle(
            symbol=symbol,
            interval_minutes=interval_minutes,
            ts_open=ts_open,
            ts_close=ts_close,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[6]),
            closed=True,
        )
