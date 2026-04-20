"""Chronological baseline backtest service.

This is an inspectable v1 simulator for validation gates, not a full exchange
fill model. It keeps fee/slippage accounting explicit for teaching and review.
"""

from __future__ import annotations

import math

import pandas as pd

from quant_mcp.adapters.persistence.json_store import JsonStore
from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.domain.validation import BacktestMetrics, BacktestRequest, BacktestResult
from quant_mcp.enums import ValidationStatus
from quant_mcp.settings import AppSettings


class BacktestService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.frames = ParquetStore(settings.data_dir)
        self.results = JsonStore(settings.artifact_dir)

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        frame = self.frames.read_frame(f"features/{request.dataset_id}_features.parquet").copy()
        if frame.empty:
            raise ValueError("Feature table is empty")

        frame["position"] = frame["signal_trend_up"].astype(int)
        frame["gross_return"] = frame["position"] * frame["ret_1"]
        cost = (request.fee_bps + request.slippage_bps) / 10_000
        # Transaction costs are charged only when the target position changes.
        frame["trade_change"] = frame["position"].diff().abs().fillna(frame["position"].abs())
        frame["net_return"] = frame["gross_return"] - frame["trade_change"] * cost
        frame["equity"] = (1 + frame["net_return"]).cumprod()
        frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1

        trades = int(frame["trade_change"].sum())
        total_return_pct = float((frame["equity"].iloc[-1] - 1) * 100)
        benchmark_return_pct = float(((1 + frame["ret_1"]).cumprod().iloc[-1] - 1) * 100)
        wins = int((frame["net_return"] > 0).sum())
        losses = max(int((frame["net_return"] < 0).sum()), 1)
        gross_profit = float(frame.loc[frame["net_return"] > 0, "net_return"].sum())
        # Avoid division instability in tiny samples where no losing bar exists.
        gross_loss = abs(float(frame.loc[frame["net_return"] < 0, "net_return"].sum())) or 1e-9

        metrics = BacktestMetrics(
            trades=trades,
            total_return_pct=round(total_return_pct, 4),
            max_drawdown_pct=round(float(frame["drawdown"].min() * 100), 4),
            win_rate_pct=round((wins / max(wins + losses, 1)) * 100, 4),
            profit_factor=round(gross_profit / gross_loss, 4) if math.isfinite(gross_profit / gross_loss) else 0.0,
            benchmark_return_pct=round(benchmark_return_pct, 4),
        )
        status = (
            ValidationStatus.PASS
            if metrics.total_return_pct > metrics.benchmark_return_pct
            else ValidationStatus.WARNING
        )
        result = BacktestResult(
            strategy_id=request.strategy_id,
            dataset_id=request.dataset_id,
            status=status,
            metrics=metrics,
            notes="Simple baseline event-driven backtest. Chronological only; no random shuffles.",
        )
        self.results.write_model(f"backtests/{result.run_id}.json", result)
        return result

    def compare_backtests(self) -> list[BacktestResult]:
        root = self.settings.artifact_dir / "backtests"
        if not root.exists():
            return []
        return [self.results.read_model(f"backtests/{p.name}", BacktestResult) for p in root.glob("*.json")]
