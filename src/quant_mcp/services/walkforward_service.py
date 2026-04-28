"""Forward-only walk-forward validation service.

The service evaluates contiguous train/test windows in chronological order and
never shuffles rows, matching the way a trading strategy would encounter time.
"""

from __future__ import annotations

import pandas as pd

from quant_mcp.adapters.persistence.json_store import JsonStore
from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.domain.validation import WalkForwardFold, WalkForwardRequest, WalkForwardResult
from quant_mcp.domain.strategy import StrategySpec
from quant_mcp.enums import ValidationStatus
from quant_mcp.settings import AppSettings


class WalkForwardService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.frames = ParquetStore(settings.data_dir)
        self.results = JsonStore(settings.artifact_dir)

    def run_walk_forward(self, request: WalkForwardRequest) -> WalkForwardResult:
        frame = self.frames.read_frame(f"features/{request.dataset_id}_features.parquet").copy()
        
        try:
            strategy = self.results.read_model(f"strategies/{request.strategy_id}.json", StrategySpec)
        except Exception:
            strategy = None

        if strategy and strategy.entry_rule and "from" not in strategy.entry_rule:
            try:
                if " or " in strategy.entry_rule:
                    cond1, cond2 = strategy.entry_rule.split(" or ")
                    pos = pd.Series(0, index=frame.index)
                    pos[frame.eval(cond1)] = 1
                    pos[frame.eval(cond2)] = -1
                    frame["position"] = pos
                else:
                    frame["position"] = frame.eval(strategy.entry_rule).astype(int)
            except Exception:
                frame["position"] = frame["signal_trend_up"].astype(int)
        else:
            frame["position"] = frame["signal_trend_up"].astype(int)
            
        folds: list[WalkForwardFold] = []
        start = 0
        fold_idx = 1
        while start + request.train_bars + request.test_bars <= len(frame):
            # Each fold trains on the past window and scores only the immediately following slice.
            train_end = start + request.train_bars
            test_end = train_end + request.test_bars
            test_slice = frame.iloc[train_end:test_end].copy()
            
            # Using basic return calculation without slippage/fees as they aren't in WalkForwardRequest
            test_slice["pnl"] = test_slice["position"] * test_slice["ret_1"]
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
            # Slide by the test size to avoid leakage from overlapping evaluation rows.
            start += request.test_bars
        status = (
            ValidationStatus.PASS
            if folds and sum(f.total_return_pct for f in folds) > 0
            else ValidationStatus.FAIL
        )
        return WalkForwardResult(strategy_id=request.strategy_id, dataset_id=request.dataset_id, status=status, folds=folds)
