"""Validation result contracts for backtests, walk-forward tests, and forward gates."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from quant_mcp.domain.common import utc_now
from quant_mcp.enums import ValidationStatus


class BacktestRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    benchmark_symbol: str | None = None


class BacktestMetrics(BaseModel):
    trades: int
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    benchmark_return_pct: float


class BacktestResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    dataset_id: str
    status: ValidationStatus
    metrics: BacktestMetrics
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None


class WalkForwardRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    train_bars: int = 200
    test_bars: int = 50


class WalkForwardFold(BaseModel):
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    total_return_pct: float


class WalkForwardResult(BaseModel):
    strategy_id: str
    dataset_id: str
    status: ValidationStatus
    folds: list[WalkForwardFold]
    created_at: datetime = Field(default_factory=utc_now)


class ForwardTestRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    bars: int = 50


class ForwardTestResult(BaseModel):
    strategy_id: str
    dataset_id: str
    status: ValidationStatus
    paper_path_ready: bool
    notes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
