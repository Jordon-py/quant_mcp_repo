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
        frame["signal_trend_up"] = (frame["ma_fast"] > frame["ma_slow"]).astype(int)
        frame = frame.shift(1)
        frame = frame.dropna().reset_index(drop=True)
        path = self.store.write_frame(f"features/{request.dataset_id}_features.parquet", frame)
        return FeatureTableResult(
            dataset_id=request.dataset_id,
            feature_path=str(path),
            rows=len(frame),
            columns=list(frame.columns),
        )
