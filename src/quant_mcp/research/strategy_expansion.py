"""Critique-driven BTC/SOL strategy research workflow.

This module evaluates deterministic, inspectable strategies from local Kraken
datasets. Signals are shifted before returns are applied so every backtest uses
only information available before the evaluated bar.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_mcp.adapters.persistence.parquet_store import ParquetStore
from quant_mcp.research.backtest_engine import (
    PerformanceMetrics,
    apply_delayed_exposure,
    cagr_pct as engine_cagr_pct,
    net_strategy_returns as engine_net_strategy_returns,
    performance_metrics as engine_performance_metrics,
    profit_factor as engine_profit_factor,
    sharpe_ratio as engine_sharpe_ratio,
    trade_returns as engine_trade_returns,
)
from quant_mcp.services.dataset_service import DatasetService
from quant_mcp.settings import AppSettings


StrategyBuilder = Callable[[pd.DataFrame, dict[str, pd.DataFrame]], pd.Series]


@dataclass(frozen=True)
class StrategyBlueprint:
    strategy_id: str
    name: str
    asset: str
    style: str
    hypothesis: str
    market_regime_fit: str
    feature_set: list[str]
    entry_logic: str
    exit_logic: str
    stop_risk_logic: str
    position_sizing: str
    expected_strengths: list[str]
    expected_weaknesses: list[str]
    failure_modes: list[str]
    baseline_difference: str
    builder_name: str


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    net_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    trades: int


@dataclass(frozen=True)
class StrategyEvaluation:
    blueprint: StrategyBlueprint
    full_period: PerformanceMetrics
    train: PerformanceMetrics
    validation: PerformanceMetrics
    test: PerformanceMetrics
    walk_forward: list[WalkForwardFoldResult]
    walk_forward_total_return_pct: float
    walk_forward_positive_fold_rate_pct: float
    robustness_score: float
    deployability_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchRunResult:
    created_at: str
    baseline_report_path: str
    json_path: str
    markdown_path: str
    baseline_critique: str
    evaluations: list[StrategyEvaluation]


def dataset_id_for(symbol: str, interval_minutes: int) -> str:
    return DatasetService.dataset_id(symbol, interval_minutes)


def load_latest_baseline(repo_root: Path) -> tuple[Path, dict]:
    baseline_dir = repo_root / "artifacts" / "trend_experiments"
    reports = sorted(baseline_dir.glob("trend_experiment_*.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(
            "No baseline trend experiment found. Run "
            "`python -m quant_mcp.experiments.trend_backtest` first."
        )
    path = reports[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def load_market_frame(settings: AppSettings, symbol: str, interval_minutes: int) -> pd.DataFrame:
    store = ParquetStore(settings.data_dir)
    dataset_id = dataset_id_for(symbol, interval_minutes)
    frame = store.read_frame(f"datasets/{dataset_id}.parquet")
    if frame.empty:
        raise ValueError(f"Dataset {dataset_id} is empty")
    return prepare_features(frame)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ts_open"] = pd.to_datetime(out["ts_open"], utc=True)
    out = out.sort_values("ts_open").drop_duplicates("ts_open").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = out[column].astype(float)

    out["ret_1"] = out["close"].pct_change().fillna(0)
    out["log_ret_1"] = np.log(out["close"]).diff().fillna(0)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_50"] = out["close"].rolling(50).mean()
    out["ma_100"] = out["close"].rolling(100).mean()
    out["momentum_6"] = out["close"].pct_change(6)
    out["momentum_12"] = out["close"].pct_change(12)
    out["momentum_24"] = out["close"].pct_change(24)
    out["rolling_high_24"] = out["high"].shift(1).rolling(24).max()
    out["rolling_high_48"] = out["high"].shift(1).rolling(48).max()
    out["rolling_low_12"] = out["low"].shift(1).rolling(12).min()
    out["rolling_low_24"] = out["low"].shift(1).rolling(24).min()
    out["volatility_24"] = out["ret_1"].rolling(24).std()
    out["volatility_median_100"] = out["volatility_24"].rolling(100).median()
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma_20"]
    out["atr_14"] = average_true_range(out, 14)
    out["atr_pct"] = out["atr_14"] / out["close"]
    out["atr_pct_median_100"] = out["atr_pct"].rolling(100).median()
    out["rsi_14"] = rsi(out["close"], 14)
    out["zscore_20"] = (out["close"] - out["ma_20"]) / out["close"].rolling(20).std()
    return out.dropna().reset_index(drop=True)


def average_true_range(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_btc_context(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = frame.copy()
    source["_source_index"] = source.index
    btc = context["BTC/USD"][
        ["ts_open", "close", "ma_20", "ma_50", "ma_100", "momentum_24", "ret_1"]
    ].rename(
        columns={
            "close": "btc_close",
            "ma_20": "btc_ma_20",
            "ma_50": "btc_ma_50",
            "ma_100": "btc_ma_100",
            "momentum_24": "btc_momentum_24",
            "ret_1": "btc_ret_1",
        }
    )
    merged = source.merge(btc, on="ts_open", how="inner")
    merged["sol_btc_ratio"] = merged["close"] / merged["btc_close"]
    merged["sol_btc_ma_20"] = merged["sol_btc_ratio"].rolling(20).mean()
    merged["sol_btc_ma_50"] = merged["sol_btc_ratio"].rolling(50).mean()
    merged["sol_btc_momentum_12"] = merged["sol_btc_ratio"].pct_change(12)
    return merged.dropna().set_index("_source_index", drop=True)


def stateful_position(
    frame: pd.DataFrame,
    entry: pd.Series,
    exit_signal: pd.Series,
    *,
    max_hold_bars: int | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.Series:
    in_position = False
    entry_price = 0.0
    hold_bars = 0
    positions: list[int] = []
    closes = frame["close"].to_numpy()

    for idx, close in enumerate(closes):
        if not in_position and bool(entry.iloc[idx]):
            in_position = True
            entry_price = float(close)
            hold_bars = 0
        elif in_position:
            hold_bars += 1
            stop_hit = stop_loss_pct is not None and float(close) <= entry_price * (1 - stop_loss_pct)
            target_hit = take_profit_pct is not None and float(close) >= entry_price * (1 + take_profit_pct)
            max_hold_hit = max_hold_bars is not None and hold_bars >= max_hold_bars
            if bool(exit_signal.iloc[idx]) or stop_hit or target_hit or max_hold_hit:
                in_position = False
        positions.append(1 if in_position else 0)

    return pd.Series(positions, index=frame.index, dtype=float)


def build_btc_regime_trend(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    vol_ok = frame["volatility_24"] < frame["volatility_median_100"] * 1.35
    entry = (
        (frame["close"] > frame["ma_100"])
        & (frame["ma_20"] > frame["ma_50"])
        & (frame["momentum_24"] > 0)
        & vol_ok
    )
    exit_signal = (frame["close"] < frame["ma_20"]) | (frame["ma_20"] < frame["ma_50"])
    return stateful_position(frame, entry, exit_signal, max_hold_bars=72, stop_loss_pct=0.035)


def build_btc_volume_breakout(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    atr_expanding = frame["atr_pct"] > frame["atr_pct_median_100"] * 1.05
    entry = (
        (frame["close"] > frame["rolling_high_48"])
        & (frame["volume_ratio"] > 1.15)
        & atr_expanding
        & (frame["close"] > frame["ma_50"])
    )
    exit_signal = (frame["close"] < frame["ma_20"]) | (frame["close"] < frame["rolling_low_12"])
    return stateful_position(frame, entry, exit_signal, max_hold_bars=36, stop_loss_pct=0.03)


def build_btc_pullback_reversion(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    regime = (frame["close"] > frame["ma_100"]) & (frame["ma_20"] > frame["ma_50"])
    entry = regime & ((frame["rsi_14"] < 38) | (frame["zscore_20"] < -1.35))
    exit_signal = (frame["rsi_14"] > 55) | (frame["close"] > frame["ma_20"]) | (frame["close"] < frame["ma_50"])
    return stateful_position(frame, entry, exit_signal, max_hold_bars=24, stop_loss_pct=0.025)


def build_sol_relative_strength_momentum(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    work = add_btc_context(frame, context)
    risk_on = (work["btc_close"] > work["btc_ma_50"]) & (work["btc_momentum_24"] > -0.01)
    entry = (
        risk_on
        & (work["close"] > work["ma_50"])
        & (work["momentum_12"] > 0.025)
        & (work["sol_btc_ratio"] > work["sol_btc_ma_20"])
        & (work["volume_ratio"] > 1.0)
    )
    exit_signal = (
        (work["sol_btc_ratio"] < work["sol_btc_ma_20"])
        | (work["close"] < work["ma_20"])
        | (work["btc_close"] < work["btc_ma_50"])
    )
    position = stateful_position(work, entry, exit_signal, max_hold_bars=36, stop_loss_pct=0.055)
    return position.reindex(frame.index).fillna(0)


def build_sol_volume_breakout(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    work = add_btc_context(frame, context)
    risk_on = work["btc_close"] > work["btc_ma_100"]
    entry = (
        risk_on
        & (work["close"] > work["rolling_high_24"])
        & (work["volume_ratio"] > 1.45)
        & (work["atr_pct"] > work["atr_pct_median_100"])
    )
    exit_signal = (work["close"] < work["ma_10"]) | (work["rsi_14"] < 48)
    position = stateful_position(work, entry, exit_signal, max_hold_bars=18, stop_loss_pct=0.06)
    return position.reindex(frame.index).fillna(0)


def build_sol_flush_reversion(frame: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    work = add_btc_context(frame, context)
    btc_not_broken = (work["btc_close"] > work["btc_ma_100"]) | (work["btc_momentum_24"] > -0.02)
    flush = (work["ret_1"] < -0.04) | ((work["zscore_20"] < -1.8) & (work["rsi_14"] < 35))
    entry = btc_not_broken & flush & (work["volume_ratio"] > 1.1)
    exit_signal = (work["rsi_14"] > 52) | (work["close"] > work["ma_10"]) | (work["btc_momentum_24"] < -0.04)
    position = stateful_position(
        work,
        entry,
        exit_signal,
        max_hold_bars=18,
        stop_loss_pct=0.07,
        take_profit_pct=0.08,
    )
    return position.reindex(frame.index).fillna(0)


BUILDERS: dict[str, StrategyBuilder] = {
    "btc_regime_trend_guard": build_btc_regime_trend,
    "btc_volume_breakout": build_btc_volume_breakout,
    "btc_pullback_reversion": build_btc_pullback_reversion,
    "sol_relative_strength_momentum": build_sol_relative_strength_momentum,
    "sol_volume_breakout": build_sol_volume_breakout,
    "sol_flush_reversion": build_sol_flush_reversion,
}


def strategy_blueprints() -> list[StrategyBlueprint]:
    return [
        StrategyBlueprint(
            strategy_id="btc_regime_trend_guard",
            name="BTC Regime Trend Guard",
            asset="BTC/USD",
            style="trend-following / regime filter",
            hypothesis="BTC trends are more durable when price is above long-term trend and short trend leads medium trend.",
            market_regime_fit="Persistent uptrends with moderate volatility.",
            feature_set=["ma_20", "ma_50", "ma_100", "momentum_24", "volatility_24"],
            entry_logic="Long when close > ma_100, ma_20 > ma_50, 24h momentum > 0, and volatility is not extreme.",
            exit_logic="Exit on close below ma_20, ma_20 below ma_50, 72-bar max hold, or stop.",
            stop_risk_logic="3.5% stop with fee/slippage charged on entries and exits.",
            position_sizing="Binary research exposure; production version should volatility-target size.",
            expected_strengths=["Filters weak BTC chop", "Avoids high-volatility late entries"],
            expected_weaknesses=["Can miss V-shaped reversals", "Slow in regime transitions"],
            failure_modes=["Sideways whipsaw", "Sudden macro selloff", "Volatility compression before false trend"],
            baseline_difference="Adds regime and volatility filters to the baseline MA signal.",
            builder_name="btc_regime_trend_guard",
        ),
        StrategyBlueprint(
            strategy_id="btc_volume_breakout",
            name="BTC Volume Confirmed Breakout",
            asset="BTC/USD",
            style="breakout / volatility expansion",
            hypothesis="BTC breakouts have better follow-through when they clear prior range highs on real volume and ATR expansion.",
            market_regime_fit="Range expansion after consolidation.",
            feature_set=["rolling_high_48", "volume_ratio", "atr_pct", "ma_50"],
            entry_logic="Long on close above prior 48h high with volume_ratio > 1.15, ATR expansion, and close > ma_50.",
            exit_logic="Exit on close below ma_20, close below prior 12h low, 36-bar max hold, or stop.",
            stop_risk_logic="3% stop to avoid failed-breakout bleed.",
            position_sizing="Binary research exposure; production version should reduce size when ATR is large.",
            expected_strengths=["Avoids weak MA-only trend entries", "Responds to expansion regimes"],
            expected_weaknesses=["Few trades", "False breakouts can cluster"],
            failure_modes=["News wicks", "Breakout exhaustion", "Low-liquidity fakeouts"],
            baseline_difference="Requires structure break plus volume instead of only moving-average alignment.",
            builder_name="btc_volume_breakout",
        ),
        StrategyBlueprint(
            strategy_id="btc_pullback_reversion",
            name="BTC Uptrend Pullback Reversion",
            asset="BTC/USD",
            style="mean reversion inside trend",
            hypothesis="BTC pullbacks in confirmed uptrends can rebound without needing a new breakout.",
            market_regime_fit="Constructive uptrends with temporary oversold dips.",
            feature_set=["ma_20", "ma_50", "ma_100", "rsi_14", "zscore_20"],
            entry_logic="Long when long-term trend is up and RSI/z-score show a pullback.",
            exit_logic="Exit on RSI normalization, close above ma_20, close below ma_50, 24-bar max hold, or stop.",
            stop_risk_logic="2.5% stop because failed pullbacks can become trend breaks.",
            position_sizing="Binary research exposure; production version should size smaller during elevated ATR.",
            expected_strengths=["Can enter earlier than breakout systems", "Designed for BTC trend pullbacks"],
            expected_weaknesses=["Catches falling knives if regime breaks", "May exit before large continuation"],
            failure_modes=["Trend reversal", "High-volatility liquidation cascade", "Oversold stays oversold"],
            baseline_difference="Uses contrarian timing inside a trend instead of chasing MA confirmation.",
            builder_name="btc_pullback_reversion",
        ),
        StrategyBlueprint(
            strategy_id="sol_relative_strength_momentum",
            name="SOL Relative Strength Momentum",
            asset="SOL/USD",
            style="momentum confirmation / BTC risk filter",
            hypothesis="SOL momentum is more reliable when SOL is outperforming BTC while BTC is not in a risk-off regime.",
            market_regime_fit="Risk-on altcoin rotations where SOL leads BTC.",
            feature_set=["momentum_12", "sol_btc_ratio", "sol_btc_ma_20", "btc_ma_50", "volume_ratio"],
            entry_logic="Long SOL when BTC is risk-on, SOL is above ma_50, 12h momentum is positive, and SOL/BTC is above its 20h mean.",
            exit_logic="Exit when SOL/BTC weakens, SOL loses ma_20, BTC loses ma_50, 36-bar max hold, or stop.",
            stop_risk_logic="5.5% stop to account for SOL volatility.",
            position_sizing="Binary research exposure; production version should cap SOL allocation below BTC.",
            expected_strengths=["Uses BTC context", "Targets SOL-specific leadership"],
            expected_weaknesses=["Depends on aligned BTC/SOL timestamps", "Can lag early SOL reversals"],
            failure_modes=["BTC risk-off shock", "SOL-specific news gap", "Relative strength whipsaw"],
            baseline_difference="Adds cross-asset confirmation that the baseline SOL trend strategy lacked.",
            builder_name="sol_relative_strength_momentum",
        ),
        StrategyBlueprint(
            strategy_id="sol_volume_breakout",
            name="SOL Volume Expansion Breakout",
            asset="SOL/USD",
            style="breakout / volume expansion",
            hypothesis="SOL breakouts need strong volume and risk-on BTC context to survive high noise.",
            market_regime_fit="Altcoin expansion bursts after compressed ranges.",
            feature_set=["rolling_high_24", "volume_ratio", "atr_pct", "btc_ma_100", "rsi_14"],
            entry_logic="Long when SOL closes above prior 24h high with volume_ratio > 1.45, ATR expansion, and BTC above ma_100.",
            exit_logic="Exit on close below ma_10, RSI below 48, 18-bar max hold, or stop.",
            stop_risk_logic="6% stop because SOL breakouts are volatile and false moves can reverse quickly.",
            position_sizing="Binary research exposure; production version should size by ATR and liquidity.",
            expected_strengths=["Demands real participation", "Shorter hold avoids stale altcoin exposure"],
            expected_weaknesses=["May trade infrequently", "Volume spikes can mark exhaustion"],
            failure_modes=["Breakout trap", "BTC turns risk-off", "Post-spike liquidity fade"],
            baseline_difference="Uses structure, volume, and BTC context instead of simple MA trend.",
            builder_name="sol_volume_breakout",
        ),
        StrategyBlueprint(
            strategy_id="sol_flush_reversion",
            name="SOL Flush Reversion With BTC Guard",
            asset="SOL/USD",
            style="mean reversion / risk guard",
            hypothesis="Sharp SOL flushes can mean-revert if BTC has not structurally broken.",
            market_regime_fit="Risk-on or neutral markets with temporary SOL liquidation spikes.",
            feature_set=["ret_1", "zscore_20", "rsi_14", "volume_ratio", "btc_momentum_24"],
            entry_logic="Long after sharp SOL downside flush or oversold z-score when BTC is not broken and volume confirms capitulation.",
            exit_logic="Exit on RSI recovery, close above ma_10, BTC momentum break, 18-bar max hold, stop, or target.",
            stop_risk_logic="7% stop and 8% take-profit because this is a high-volatility reversion trade.",
            position_sizing="Binary research exposure; production version should be smallest of the six strategies.",
            expected_strengths=["Different payoff profile from trend/breakout", "Uses BTC guard against systemic selloff"],
            expected_weaknesses=["Hardest to execute live due to speed", "Can be hurt by cascading liquidations"],
            failure_modes=["SOL-specific breakdown", "BTC crash after entry", "Exchange liquidity gap"],
            baseline_difference="Trades panic reversions instead of momentum continuation.",
            builder_name="sol_flush_reversion",
        ),
    ]


def evaluate_strategy(
    blueprint: StrategyBlueprint,
    frames: dict[str, pd.DataFrame],
    *,
    fee_bps: float,
    slippage_bps: float,
    interval_minutes: int,
) -> StrategyEvaluation:
    frame = frames[blueprint.asset].copy().reset_index(drop=True)
    raw_position = BUILDERS[blueprint.builder_name](frame, frames).reindex(frame.index).fillna(0)
    backtest_frame = apply_delayed_exposure(frame, raw_position)
    net_returns = net_strategy_returns(backtest_frame, fee_bps=fee_bps, slippage_bps=slippage_bps)
    full_metrics = performance_metrics(backtest_frame, net_returns, interval_minutes)

    train_idx, validation_idx, test_idx = chronological_split_indices(len(backtest_frame))
    train_metrics = metrics_for_indices(backtest_frame, net_returns, train_idx, interval_minutes)
    validation_metrics = metrics_for_indices(backtest_frame, net_returns, validation_idx, interval_minutes)
    test_metrics = metrics_for_indices(backtest_frame, net_returns, test_idx, interval_minutes)
    folds = walk_forward_metrics(backtest_frame, net_returns, train_bars=200, test_bars=50, interval_minutes=interval_minutes)
    wf_total = round(sum(fold.net_return_pct for fold in folds), 4)
    wf_positive = round((sum(1 for fold in folds if fold.net_return_pct > 0) / max(len(folds), 1)) * 100, 4)
    robustness = robustness_score(full_metrics, validation_metrics, test_metrics, folds)
    return StrategyEvaluation(
        blueprint=blueprint,
        full_period=full_metrics,
        train=train_metrics,
        validation=validation_metrics,
        test=test_metrics,
        walk_forward=folds,
        walk_forward_total_return_pct=wf_total,
        walk_forward_positive_fold_rate_pct=wf_positive,
        robustness_score=robustness,
        deployability_notes=deployability_notes(full_metrics, validation_metrics, test_metrics, folds),
    )


def net_strategy_returns(frame: pd.DataFrame, *, fee_bps: float, slippage_bps: float) -> pd.Series:
    return engine_net_strategy_returns(frame, fee_bps=fee_bps, slippage_bps=slippage_bps)


def performance_metrics(frame: pd.DataFrame, net_returns: pd.Series, interval_minutes: int) -> PerformanceMetrics:
    return engine_performance_metrics(frame, net_returns, interval_minutes)


def cagr_pct(final_equity: float, timestamps: pd.Series) -> float:
    return engine_cagr_pct(final_equity, timestamps)


def sharpe_ratio(net_returns: pd.Series, interval_minutes: int) -> float:
    return engine_sharpe_ratio(net_returns, interval_minutes)


def trade_returns(frame: pd.DataFrame, net_returns: pd.Series) -> list[float]:
    return engine_trade_returns(frame, net_returns)


def profit_factor(trades: list[float]) -> float:
    return engine_profit_factor(trades)


def chronological_split_indices(length: int) -> tuple[range, range, range]:
    train_end = int(length * 0.6)
    validation_end = int(length * 0.8)
    return range(0, train_end), range(train_end, validation_end), range(validation_end, length)


def metrics_for_indices(
    frame: pd.DataFrame,
    net_returns: pd.Series,
    indices: range,
    interval_minutes: int,
) -> PerformanceMetrics:
    subset = frame.iloc[list(indices)].copy().reset_index(drop=True)
    subset_returns = net_returns.iloc[list(indices)].reset_index(drop=True)
    if subset.empty:
        raise ValueError("Chronological split produced an empty subset")
    return performance_metrics(subset, subset_returns, interval_minutes)


def walk_forward_metrics(
    frame: pd.DataFrame,
    net_returns: pd.Series,
    *,
    train_bars: int,
    test_bars: int,
    interval_minutes: int,
) -> list[WalkForwardFoldResult]:
    folds: list[WalkForwardFoldResult] = []
    start = 0
    fold = 1
    while start + train_bars + test_bars <= len(frame):
        train_start = start
        train_end = start + train_bars
        test_start = train_end
        test_end = train_end + test_bars
        test_slice = frame.iloc[test_start:test_end].copy().reset_index(drop=True)
        test_returns = net_returns.iloc[test_start:test_end].reset_index(drop=True)
        metrics = performance_metrics(test_slice, test_returns, interval_minutes)
        folds.append(
            WalkForwardFoldResult(
                fold=fold,
                train_start=str(frame["ts_open"].iloc[train_start]),
                train_end=str(frame["ts_open"].iloc[train_end - 1]),
                test_start=str(frame["ts_open"].iloc[test_start]),
                test_end=str(frame["ts_open"].iloc[test_end - 1]),
                net_return_pct=metrics.net_return_pct,
                max_drawdown_pct=metrics.max_drawdown_pct,
                sharpe=metrics.sharpe,
                trades=metrics.trades,
            )
        )
        fold += 1
        start += test_bars
    return folds


def robustness_score(
    full_metrics: PerformanceMetrics,
    validation_metrics: PerformanceMetrics,
    test_metrics: PerformanceMetrics,
    folds: list[WalkForwardFoldResult],
) -> float:
    positive_fold_rate = sum(1 for fold in folds if fold.net_return_pct > 0) / max(len(folds), 1)
    score = (
        full_metrics.sharpe * 12
        + validation_metrics.net_return_pct
        + test_metrics.net_return_pct * 1.5
        + positive_fold_rate * 20
        + full_metrics.max_drawdown_pct
        + min(full_metrics.profit_factor, 5) * 3
    )
    if full_metrics.trades < 3:
        score -= 12
    if full_metrics.exposure_pct > 80:
        score -= 8
    return round(score, 4)


def deployability_notes(
    full_metrics: PerformanceMetrics,
    validation_metrics: PerformanceMetrics,
    test_metrics: PerformanceMetrics,
    folds: list[WalkForwardFoldResult],
) -> list[str]:
    notes: list[str] = []
    if full_metrics.net_return_pct <= full_metrics.benchmark_return_pct:
        notes.append("Did not beat buy-and-hold over the full period.")
    if test_metrics.net_return_pct <= 0:
        notes.append("Out-of-sample test split was not profitable.")
    if full_metrics.trades < 5:
        notes.append("Trade count is low; result may be sample-fragile.")
    if full_metrics.max_drawdown_pct < -12:
        notes.append("Drawdown is large for a first paper-trading candidate.")
    if sum(1 for fold in folds if fold.net_return_pct > 0) < max(len(folds) // 2, 1):
        notes.append("Walk-forward folds were not consistently positive.")
    return notes or ["Research candidate only; requires paper-trading evidence before approval."]


def baseline_critique_text(baseline: dict) -> str:
    rows = baseline.get("summary", [])
    top = rows[0] if rows else {}
    btc_rows = [row for row in rows if row.get("symbol") == "BTC/USD"]
    sol_rows = [row for row in rows if row.get("symbol") == "SOL/USD"]
    btc_best = btc_rows[0] if btc_rows else {}
    sol_best = sol_rows[0] if sol_rows else {}
    return (
        "The baseline is a lagged moving-average trend rule: go long when a fast moving average "
        "is above a slow moving average, then exit when the trend signal turns off. It assumes "
        "recent trend persistence is enough to overcome 10 bps fees plus 5 bps slippage. The "
        f"best baseline row was {top.get('symbol')} {top.get('lookback_fast')}:{top.get('lookback_slow')} "
        f"with {top.get('total_return_pct')}% net return, {top.get('max_drawdown_pct')}% max drawdown, "
        f"and {top.get('positive_fold_rate_pct')}% positive folds. BTC was weaker: the "
        f"best BTC baseline returned {btc_best.get('total_return_pct')}% versus "
        f"{btc_best.get('benchmark_return_pct')}% buy-and-hold, which means the filter reduced drawdown "
        "but gave up upside. SOL fit the baseline better in this sample because faster 5:20 trend "
        f"following captured higher-beta bursts; the best SOL row returned {sol_best.get('total_return_pct')}% "
        f"versus {sol_best.get('benchmark_return_pct')}% buy-and-hold. The weakness is structural: the "
        "baseline has no volume confirmation, no volatility-state awareness, no BTC risk filter for SOL, "
        "no explicit stop beyond signal reversal, and it can overfit to one lookback pair. It is likely "
        "to fail in sideways chop, failed breakouts, sharp liquidation cascades, and regime transitions "
        "where moving averages react late."
    )


def recommendation_label(item: StrategyEvaluation) -> str:
    if (
        item.robustness_score > 0
        and item.full_period.net_return_pct > 0
        and item.walk_forward_positive_fold_rate_pct >= 50
    ):
        return "paper-test candidate, not live-ready"
    if item.full_period.net_return_pct > 0:
        return "watchlist only"
    return "reject or redesign"


def generate_markdown(
    baseline_path: Path,
    baseline: dict,
    evaluations: list[StrategyEvaluation],
) -> str:
    critique = baseline_critique_text(baseline)
    ranked = sorted(evaluations, key=lambda item: item.robustness_score, reverse=True)
    best_btc = next(item for item in ranked if item.blueprint.asset == "BTC/USD")
    best_sol = next(item for item in ranked if item.blueprint.asset == "SOL/USD")
    lines = [
        "# Critique-Driven BTC/SOL Strategy Expansion",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Baseline artifact: `{baseline_path}`",
        "",
        "## 1. Baseline Strategy Critique",
        "",
        f"**Finding:** {critique}",
        "",
        "## Evidence Labels",
        "",
        "- **Facts:** artifact paths, saved strategy definitions, realized backtest metrics, and scheduled-task status.",
        "- **Assumptions:** close-to-close fills, long-only exposure, 10 bps fees plus 5 bps slippage, and hourly Kraken candles.",
        "- **Findings:** performance and stability observed in this sample only.",
        "- **Recommendations:** research actions, not live-trading approval.",
        "",
        "## 2. New Strategy Set",
        "",
    ]
    for evaluation in evaluations:
        bp = evaluation.blueprint
        lines.extend(
            [
                f"### {bp.name}",
                f"- **Asset:** {bp.asset}",
                f"- **Hypothesis:** {bp.hypothesis}",
                f"- **Regime fit:** {bp.market_regime_fit}",
                f"- **Feature set:** {', '.join(bp.feature_set)}",
                f"- **Entry:** {bp.entry_logic}",
                f"- **Exit:** {bp.exit_logic}",
                f"- **Risk/sizing:** {bp.stop_risk_logic} {bp.position_sizing}",
                f"- **Expected strengths:** {'; '.join(bp.expected_strengths)}",
                f"- **Expected weaknesses:** {'; '.join(bp.expected_weaknesses)}",
                f"- **Failure modes:** {'; '.join(bp.failure_modes)}",
                f"- **Difference from baseline:** {bp.baseline_difference}",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Backtest Results",
            "",
            "| Strategy | Asset | Net % | CAGR % | Sharpe | Max DD % | Win % | Profit Factor | Exposure % | Trades | Benchmark % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in ranked:
        m = item.full_period
        lines.append(
            f"| {item.blueprint.name} | {item.blueprint.asset} | {m.net_return_pct:.2f} | "
            f"{m.cagr_pct:.2f} | {m.sharpe:.2f} | {m.max_drawdown_pct:.2f} | "
            f"{m.win_rate_pct:.2f} | {m.profit_factor:.2f} | {m.exposure_pct:.2f} | "
            f"{m.trades} | {m.benchmark_return_pct:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Walk-Forward Findings",
            "",
            "| Strategy | WF Return % | Positive Folds % | Folds | Stability Notes |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in ranked:
        lines.append(
            f"| {item.blueprint.name} | {item.walk_forward_total_return_pct:.2f} | "
            f"{item.walk_forward_positive_fold_rate_pct:.2f} | {len(item.walk_forward)} | "
            f"{'; '.join(item.deployability_notes)} |"
        )
    lines.extend(
        [
            "",
            "## 5. Comparative Ranking",
            "",
            "| Rank | Strategy | Asset | Robustness Score | Recommendation |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for rank, item in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {item.blueprint.name} | {item.blueprint.asset} | "
            f"{item.robustness_score:.2f} | {recommendation_label(item)} |"
        )
    lines.extend(
        [
            "",
            "## 6. Best BTC Candidate",
            "",
            f"**Recommendation:** {best_btc.blueprint.name}. It ranked best among BTC strategies because its "
            f"test split returned {best_btc.test.net_return_pct:.2f}% with Sharpe {best_btc.test.sharpe:.2f} "
            f"and full-period drawdown {best_btc.full_period.max_drawdown_pct:.2f}%. It is still "
            f"`{recommendation_label(best_btc)}` because full-period net return was "
            f"{best_btc.full_period.net_return_pct:.2f}% versus "
            f"{best_btc.full_period.benchmark_return_pct:.2f}% benchmark.",
            "",
            "## 7. Best SOL Candidate",
            "",
            f"**Recommendation:** {best_sol.blueprint.name}. It ranked best among SOL strategies because its "
            f"test split returned {best_sol.test.net_return_pct:.2f}% with Sharpe {best_sol.test.sharpe:.2f} "
            f"and full-period drawdown {best_sol.full_period.max_drawdown_pct:.2f}%. It is "
            f"`{recommendation_label(best_sol)}` and requires paper-trading proof before any approval.",
            "",
            "## 8. Risks, Caveats, and Failure Modes",
            "",
            "- These results use recent Kraken public OHLC history only; they are not proof of durable edge.",
            "- CAGR is annualized from a short sample and should be treated as a comparison statistic, not a forecast.",
            "- The strategies are long-only and assume close-to-close execution with explicit fee/slippage costs.",
            "- No strategy should be approved for live trading without paper-trading evidence and risk-gate integration.",
            "",
            "## 9. Recommended Next Iteration",
            "",
            "1. Paper-trade the best BTC and SOL candidates with live-updating data.",
            "2. Add per-trade ledgers and equity/drawdown charts.",
            "3. Add a promotion gate that requires positive paper results before `prepare_live_trade_intent` can pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research(
    repo_root: Path,
    *,
    interval_minutes: int = 60,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> ResearchRunResult:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    baseline_path, baseline = load_latest_baseline(repo_root)
    frames = {
        "BTC/USD": load_market_frame(settings, "BTC/USD", interval_minutes),
        "SOL/USD": load_market_frame(settings, "SOL/USD", interval_minutes),
    }
    evaluations = [
        evaluate_strategy(
            blueprint,
            frames,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            interval_minutes=interval_minutes,
        )
        for blueprint in strategy_blueprints()
    ]
    markdown = generate_markdown(baseline_path, baseline, evaluations)
    out_dir = repo_root / "artifacts" / "research_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    markdown_path = out_dir / f"strategy_expansion_report_{stamp}.md"
    json_path = out_dir / f"strategy_expansion_report_{stamp}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_report_path": str(baseline_path),
        "baseline_critique": baseline_critique_text(baseline),
        "evaluations": [evaluation_to_dict(item) for item in evaluations],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ResearchRunResult(
        created_at=payload["created_at"],
        baseline_report_path=str(baseline_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        baseline_critique=payload["baseline_critique"],
        evaluations=evaluations,
    )


def evaluation_to_dict(evaluation: StrategyEvaluation) -> dict:
    data = asdict(evaluation)
    return data
