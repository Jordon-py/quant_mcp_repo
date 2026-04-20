from __future__ import annotations

from pathlib import Path

from quant_mcp.adapters.persistence.json_store import JsonStore
from quant_mcp.domain.strategy import (
    GenerateStrategyCandidatesRequest,
    StrategyListResult,
    StrategySpec,
)
from quant_mcp.settings import AppSettings


class StrategyService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.store = JsonStore(settings.artifact_dir)
        self.dir = settings.artifact_dir / "strategies"
        self.dir.mkdir(parents=True, exist_ok=True)

    def generate_strategy_candidates(self, request: GenerateStrategyCandidatesRequest) -> list[StrategySpec]:
        candidates: list[StrategySpec] = []
        for i in range(request.count):
            if request.family == "breakout":
                spec = StrategySpec(
                    name=f"{request.symbol} breakout candidate {i+1}",
                    family="breakout",
                    symbol=request.symbol,
                    interval_minutes=request.interval_minutes,
                    entry_rule="close > rolling_high_20 and volatility < rolling_vol_median",
                    exit_rule="stop_atr_2x or max_holding_bars",
                    sizing_rule="risk_pct / stop_distance",
                    notes="Generated from constrained grammar: breakout only, no LLM decision loop.",
                )
            elif request.family == "trend":
                spec = StrategySpec(
                    name=f"{request.symbol} trend candidate {i+1}",
                    family="trend",
                    symbol=request.symbol,
                    interval_minutes=request.interval_minutes,
                    entry_rule="ma_fast > ma_slow and ret_20 > 0",
                    exit_rule="ma_fast < ma_slow or max_holding_bars",
                    sizing_rule="vol_targeted",
                )
            else:
                spec = StrategySpec(
                    name=f"{request.symbol} mean reversion candidate {i+1}",
                    family="mean_reversion",
                    symbol=request.symbol,
                    interval_minutes=request.interval_minutes,
                    entry_rule="zscore_close < -2",
                    exit_rule="zscore_close > 0 or max_holding_bars",
                    sizing_rule="half_risk_pct",
                )
            candidates.append(spec)
        return candidates

    def save_strategy(self, spec: StrategySpec) -> Path:
        return self.store.write_model(f"strategies/{spec.strategy_id}.json", spec)

    def list_strategies(self) -> StrategyListResult:
        strategies: list[StrategySpec] = []
        for path in self.dir.glob("*.json"):
            strategies.append(self.store.read_model(f"strategies/{path.name}", StrategySpec))
        return StrategyListResult(strategies=strategies)
