"""Feature engineering service for saved market datasets.

It creates simple, explainable lagged features from persisted candles and keeps
future-leakage prevention inside the service boundary.
"""

from __future__ import annotations

import pandas as pd

from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.domain.dataset import FeatureTableRequest, FeatureTableResult
from quant_mcp.settings import AppSettings


class FeatureService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.store = ParquetStore(settings.data_dir)

    def build_feature_table(self, request: FeatureTableRequest) -> FeatureTableResult:
        frame = self.store.read_frame(f"datasets/{request.dataset_id}.parquet").copy()
        frame["close"] = frame["close"].astype(float)
        frame["ret_1"] = frame["close"].pct_change()
        frame["ma_fast"] = frame["close"].rolling(request.lookback_fast).mean()
        frame["ma_slow"] = frame["close"].rolling(request.lookback_slow).mean()
        frame["volatility"] = frame["ret_1"].rolling(request.lookback_fast).std()
        frame["rolling_high_20"] = frame["close"].rolling(20).max()
        frame["rolling_vol_median"] = frame["volatility"].rolling(request.lookback_slow).median()
        
        std_slow = frame["close"].rolling(request.lookback_slow).std()
        frame["zscore_close"] = (frame["close"] - frame["ma_slow"]) / std_slow

        frame["signal_trend_up"] = (frame["ma_fast"] > frame["ma_slow"]).astype(int)
        # Shift derived signal inputs, but keep the realized return on the bar being evaluated.
        # Backtests then apply yesterday's signal to today's return instead of leaking same-bar data.
        lagged_columns = ["ma_fast", "ma_slow", "volatility", "rolling_high_20", "rolling_vol_median", "zscore_close", "signal_trend_up"]
        frame[lagged_columns] = frame[lagged_columns].shift(1)
        frame = frame.dropna().reset_index(drop=True)
        path = self.store.write_frame(f"features/{request.dataset_id}_features.parquet", frame)
        return FeatureTableResult(
            dataset_id=request.dataset_id,
            feature_path=str(path),
            rows=len(frame),
            columns=list(frame.columns),
        )
