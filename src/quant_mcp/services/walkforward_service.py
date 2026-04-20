from __future__ import annotations

from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.domain.validation import WalkForwardFold, WalkForwardRequest, WalkForwardResult
from quant_mcp.enums import ValidationStatus
from quant_mcp.settings import AppSettings


class WalkForwardService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.frames = ParquetStore(settings.data_dir)

    def run_walk_forward(self, request: WalkForwardRequest) -> WalkForwardResult:
        frame = self.frames.read_frame(f"features/{request.dataset_id}_features.parquet").copy()
        folds: list[WalkForwardFold] = []
        start = 0
        fold_idx = 1
        while start + request.train_bars + request.test_bars <= len(frame):
            train_end = start + request.train_bars
            test_end = train_end + request.test_bars
            test_slice = frame.iloc[train_end:test_end].copy()
            test_slice["pnl"] = test_slice["signal_trend_up"] * test_slice["ret_1"]
            folds.append(
                WalkForwardFold(
                    fold=fold_idx,
                    train_start=start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    total_return_pct=round(float(test_slice["pnl"].sum() * 100), 4),
                )
            )
            fold_idx += 1
            start += request.test_bars
        status = ValidationStatus.PASS if folds and sum(f.total_return_pct for f in folds) > 0 else ValidationStatus.FAIL
        return WalkForwardResult(strategy_id=request.strategy_id, dataset_id=request.dataset_id, status=status, folds=folds)
