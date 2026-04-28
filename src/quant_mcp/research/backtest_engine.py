"""Shared backtest accounting for research and paper-trading workflows.

The engine is intentionally exchange-agnostic: callers provide closed-candle
features plus a raw exposure signal, and this module handles delayed execution,
costs, metrics, equity curves, and trade ledgers consistently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    net_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    exposure_pct: float
    trades: int
    benchmark_return_pct: float
    excess_return_pct: float
    calmar_ratio: float
    turnover_pct: float
    average_trade_bars: float


def apply_delayed_exposure(frame: pd.DataFrame, raw_signal: pd.Series) -> pd.DataFrame:
    """Convert a raw spot signal into next-bar executable long/flat exposure."""
    out = frame.copy().reset_index(drop=True)
    raw = raw_signal.reindex(out.index).fillna(0).astype(float).clip(lower=0.0, upper=1.0)
    out["raw_signal"] = raw
    out["position"] = raw.shift(1).fillna(0).clip(lower=0.0, upper=1.0)
    return out


def net_strategy_returns(frame: pd.DataFrame, *, fee_bps: float, slippage_bps: float) -> pd.Series:
    position = frame["position"].fillna(0).astype(float).clip(lower=0.0, upper=1.0)
    previous_position = position.shift(1).fillna(0)
    cost = (fee_bps + slippage_bps) / 10_000
    trade_change = (position - previous_position).abs()
    return position * frame["ret_1"].fillna(0).astype(float) - trade_change * cost


def build_equity_frame(
    frame: pd.DataFrame,
    net_returns: pd.Series,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    position = frame["position"].fillna(0).astype(float).clip(lower=0.0, upper=1.0)
    previous_position = position.shift(1).fillna(0)
    trade_change = (position - previous_position).abs()
    equity_multiple = (1 + net_returns.fillna(0)).cumprod()
    benchmark_multiple = (1 + frame["ret_1"].fillna(0).astype(float)).cumprod()
    equity = equity_multiple * initial_capital
    benchmark_equity = benchmark_multiple * initial_capital
    drawdown = equity / equity.cummax() - 1
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1

    columns = [
        "ts_open",
        "ts_close",
        "symbol",
        "interval_minutes",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1",
        "raw_signal",
        "position",
    ]
    available = [column for column in columns if column in frame.columns]
    out = frame[available].copy()
    if "raw_signal" not in out:
        out["raw_signal"] = 0.0
    out["paper_action"] = np.select(
        [
            (position > 0) & (previous_position <= 0),
            (position <= 0) & (previous_position > 0),
            position > 0,
        ],
        ["enter", "exit", "hold"],
        default="flat",
    )
    out["entry_reason"] = np.where(out["paper_action"] == "enter", "strategy_signal", "")
    out["exit_reason"] = np.where(out["paper_action"] == "exit", "strategy_exit_or_risk_rule", "")
    out["trade_change"] = trade_change
    out["net_return"] = net_returns.fillna(0)
    out["account_equity"] = equity
    out["equity_before_bar"] = equity.shift(1).fillna(initial_capital)
    out["benchmark_equity"] = benchmark_equity
    out["drawdown_pct"] = drawdown * 100
    out["benchmark_drawdown_pct"] = benchmark_drawdown * 100
    out["exposure_value"] = equity * position
    out["cash_estimate"] = equity * (1 - position)
    return out


def performance_metrics(
    frame: pd.DataFrame,
    net_returns: pd.Series,
    interval_minutes: int,
) -> PerformanceMetrics:
    if frame.empty:
        raise ValueError("Cannot compute metrics for an empty frame")
    equity = (1 + net_returns.fillna(0)).cumprod()
    benchmark = (1 + frame["ret_1"].fillna(0).astype(float)).cumprod()
    drawdown = equity / equity.cummax() - 1
    position = frame["position"].fillna(0).astype(float).clip(lower=0.0, upper=1.0)
    trades = build_trade_ledger(
        build_equity_frame(frame, net_returns, initial_capital=1.0),
        initial_capital=1.0,
    )
    trade_returns = trades["net_return_pct"].astype(float).div(100).tolist() if not trades.empty else []
    net_return = float((equity.iloc[-1] - 1) * 100)
    benchmark_return = float((benchmark.iloc[-1] - 1) * 100)
    max_drawdown = float(drawdown.min() * 100)
    return PerformanceMetrics(
        net_return_pct=round(net_return, 4),
        cagr_pct=round(cagr_pct(equity.iloc[-1], frame["ts_open"]), 4),
        sharpe=round(sharpe_ratio(net_returns, interval_minutes), 4),
        max_drawdown_pct=round(max_drawdown, 4),
        win_rate_pct=round((sum(1 for trade in trade_returns if trade > 0) / max(len(trade_returns), 1)) * 100, 4),
        profit_factor=round(profit_factor(trade_returns), 4),
        exposure_pct=round(float(position.mean() * 100), 4),
        trades=len(trade_returns),
        benchmark_return_pct=round(benchmark_return, 4),
        excess_return_pct=round(net_return - benchmark_return, 4),
        calmar_ratio=round(calmar_ratio(net_return, max_drawdown), 4),
        turnover_pct=round(float(position.diff().abs().fillna(position.abs()).sum() * 100), 4),
        average_trade_bars=round(float(trades["bars_held"].mean()) if not trades.empty else 0.0, 4),
    )


def cagr_pct(final_equity: float, timestamps: pd.Series) -> float:
    start = pd.to_datetime(timestamps.iloc[0], utc=True)
    end = pd.to_datetime(timestamps.iloc[-1], utc=True)
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
    return (float(final_equity) ** (1 / years) - 1) * 100


def sharpe_ratio(net_returns: pd.Series, interval_minutes: int) -> float:
    std = float(net_returns.std())
    if std == 0 or math.isnan(std):
        return 0.0
    periods_per_year = (365.25 * 24 * 60) / interval_minutes
    return float(net_returns.mean() / std * math.sqrt(periods_per_year))


def calmar_ratio(net_return_pct: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct >= 0:
        return 0.0
    return net_return_pct / abs(max_drawdown_pct)


def profit_factor(trades: list[float]) -> float:
    gross_profit = sum(trade for trade in trades if trade > 0)
    gross_loss = abs(sum(trade for trade in trades if trade < 0))
    if gross_loss == 0:
        return gross_profit / 1e-9 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def build_trade_ledger(equity_frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    position = equity_frame["position"].fillna(0).astype(float).clip(lower=0.0, upper=1.0).to_numpy()
    previous = pd.Series(position).shift(1).fillna(0).to_numpy()
    trades: list[dict] = []
    entry_idx: int | None = None

    for idx, current_position in enumerate(position):
        if current_position > 0 and previous[idx] <= 0:
            entry_idx = idx
        if entry_idx is not None and current_position <= 0 and previous[idx] > 0:
            trades.append(_trade_row(equity_frame, entry_idx, idx, len(trades) + 1, "closed"))
            entry_idx = None

    if entry_idx is not None:
        trades.append(_trade_row(equity_frame, entry_idx, len(equity_frame) - 1, len(trades) + 1, "open"))

    columns = [
        "trade_id",
        "status",
        "entry_ts",
        "exit_ts",
        "bars_held",
        "entry_price",
        "exit_price",
        "entry_exposure_pct",
        "quantity_estimate",
        "gross_return_pct",
        "net_return_pct",
        "entry_equity",
        "exit_equity",
        "net_pnl",
        "max_adverse_excursion_pct",
        "max_favorable_excursion_pct",
        "entry_reason",
        "exit_reason",
    ]
    return pd.DataFrame(trades, columns=columns)


def _trade_row(
    equity_frame: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    trade_id: int,
    status: str,
) -> dict:
    segment = equity_frame.iloc[entry_idx : exit_idx + 1]
    held_segment = segment[segment["position"].astype(float) > 0]
    entry = equity_frame.iloc[entry_idx]
    exit_row = equity_frame.iloc[exit_idx]
    entry_equity = float(entry["equity_before_bar"])
    exit_equity = float(exit_row["account_equity"])
    entry_price = float(entry["close"])
    exit_price = float(exit_row["close"])
    entry_exposure = float(entry["position"])
    gross_return = exit_price / entry_price - 1 if entry_price else 0.0
    net_return = exit_equity / max(entry_equity, 1e-9) - 1
    quantity = (entry_equity * entry_exposure) / entry_price if entry_price else 0.0
    lows = held_segment["low"] if not held_segment.empty else segment["low"]
    highs = held_segment["high"] if not held_segment.empty else segment["high"]
    exit_reason = "open_at_end" if status == "open" else str(exit_row.get("exit_reason", "strategy_exit_or_risk_rule"))
    if not exit_reason:
        exit_reason = "strategy_exit_or_risk_rule"

    return {
        "trade_id": trade_id,
        "status": status,
        "entry_ts": entry["ts_open"],
        "exit_ts": exit_row["ts_open"],
        "bars_held": int((held_segment["position"].astype(float) > 0).sum()) if not held_segment.empty else 0,
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "entry_exposure_pct": round(entry_exposure * 100, 4),
        "quantity_estimate": round(quantity, 8),
        "gross_return_pct": round(gross_return * 100, 4),
        "net_return_pct": round(net_return * 100, 4),
        "entry_equity": round(entry_equity, 2),
        "exit_equity": round(exit_equity, 2),
        "net_pnl": round(exit_equity - entry_equity, 2),
        "max_adverse_excursion_pct": round((float(lows.min()) / entry_price - 1) * 100, 4) if entry_price else 0.0,
        "max_favorable_excursion_pct": round((float(highs.max()) / entry_price - 1) * 100, 4) if entry_price else 0.0,
        "entry_reason": str(entry.get("entry_reason", "strategy_signal")) or "strategy_signal",
        "exit_reason": exit_reason,
    }


def trade_returns(frame: pd.DataFrame, net_returns: pd.Series) -> list[float]:
    ledger = build_trade_ledger(build_equity_frame(frame, net_returns, initial_capital=1.0), initial_capital=1.0)
    if ledger.empty:
        return []
    return ledger["net_return_pct"].astype(float).div(100).tolist()

