"""Strategy specification contracts.

These models describe deterministic strategy hypotheses; they are not executable
strategy code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from quant_mcp.domain.common import utc_now
from quant_mcp.enums import StrategyStatus


class StrategyConstraints(BaseModel):
    max_holding_bars: int = 24
    risk_per_trade_pct: float = 0.01
    allowed_sides: list[Literal["long", "short"]] = Field(default_factory=lambda: ["long", "short"])
    require_volume_filter: bool = True


class StrategySpec(BaseModel):
    strategy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    family: Literal["breakout", "trend", "mean_reversion"]
    symbol: str
    interval_minutes: int = 60
    entry_rule: str
    exit_rule: str
    sizing_rule: str
    constraints: StrategyConstraints = Field(default_factory=StrategyConstraints)
    status: StrategyStatus = StrategyStatus.DRAFT
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None


class GenerateStrategyCandidatesRequest(BaseModel):
    symbol: str
    interval_minutes: int = 60
    family: Literal["breakout", "trend", "mean_reversion"] = "breakout"
    count: int = 3


class StrategyListResult(BaseModel):
    strategies: list[StrategySpec]
