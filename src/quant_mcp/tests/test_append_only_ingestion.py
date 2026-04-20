import pandas as pd

from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.settings import AppSettings


def test_append_only_dedupe_shape(tmp_path):
    settings = AppSettings(data_dir=tmp_path / "data")
    store = ParquetStore(settings.data_dir)
    existing = pd.DataFrame(
        [
            {"symbol": "BTC/USD", "interval_minutes": 60, "ts_open": "2026-01-01T00:00:00Z", "close": 1, "closed": True},
            {"symbol": "BTC/USD", "interval_minutes": 60, "ts_open": "2026-01-01T01:00:00Z", "close": 2, "closed": True},
        ]
    )
    incoming = pd.DataFrame(
        [
            {"symbol": "BTC/USD", "interval_minutes": 60, "ts_open": "2026-01-01T01:00:00Z", "close": 2, "closed": True},
            {"symbol": "BTC/USD", "interval_minutes": 60, "ts_open": "2026-01-01T02:00:00Z", "close": 3, "closed": True},
        ]
    )
    combined = pd.concat([existing, incoming], ignore_index=True).drop_duplicates(
        subset=["symbol", "interval_minutes", "ts_open"], keep="first"
    )
    path = store.write_frame("datasets/test.parquet", combined)
    out = store.read_frame("datasets/test.parquet")
    assert len(out) == 3
