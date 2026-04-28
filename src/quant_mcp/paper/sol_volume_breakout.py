"""Paper ledger for the SOL Volume Expansion Breakout strategy.

This module reuses the corrected research signal, applies one-bar delayed
execution, charges fee/slippage on position changes, and exports inspectable
paper-trading artifacts. It is intentionally historical/paper-only.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_mcp.research.strategy_expansion import (
    BUILDERS,
    PerformanceMetrics,
    load_market_frame,
    net_strategy_returns,
    performance_metrics,
    strategy_blueprints,
)
from quant_mcp.settings import AppSettings


STRATEGY_ID = "sol_volume_breakout"


@dataclass(frozen=True)
class PaperLedgerResult:
    strategy_id: str
    strategy_name: str
    created_at: str
    summary_path: str
    paper_ledger_csv: str
    equity_curve_csv: str
    trades_csv: str
    equity_chart_svg: str
    drawdown_chart_svg: str
    metrics: PerformanceMetrics
    latest_state: dict


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def selected_blueprint():
    return next(item for item in strategy_blueprints() if item.strategy_id == STRATEGY_ID)


def build_paper_frame(
    settings: AppSettings,
    *,
    interval_minutes: int,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> tuple[pd.DataFrame, PerformanceMetrics]:
    frames = {
        "BTC/USD": load_market_frame(settings, "BTC/USD", interval_minutes),
        "SOL/USD": load_market_frame(settings, "SOL/USD", interval_minutes),
    }
    blueprint = selected_blueprint()
    frame = frames[blueprint.asset].copy().reset_index(drop=True)
    raw_signal = BUILDERS[blueprint.builder_name](frame, frames).reindex(frame.index).fillna(0)
    position = raw_signal.shift(1).fillna(0).clip(lower=0, upper=1)
    backtest_frame = frame.assign(raw_signal=raw_signal, position=position)
    net_returns = net_strategy_returns(backtest_frame, fee_bps=fee_bps, slippage_bps=slippage_bps)
    metrics = performance_metrics(backtest_frame, net_returns, interval_minutes)
    return build_equity_frame(backtest_frame, net_returns, initial_capital), metrics


def build_equity_frame(
    frame: pd.DataFrame,
    net_returns: pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    position = frame["position"].fillna(0).astype(float)
    previous_position = position.shift(1).fillna(0)
    trade_change = (position - previous_position).abs()
    equity_multiple = (1 + net_returns.fillna(0)).cumprod()
    benchmark_multiple = (1 + frame["ret_1"].fillna(0)).cumprod()
    equity = equity_multiple * initial_capital
    benchmark_equity = benchmark_multiple * initial_capital
    drawdown = equity / equity.cummax() - 1
    benchmark_drawdown = benchmark_equity / benchmark_equity.cummax() - 1

    out = frame[
        [
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
    ].copy()
    out["paper_action"] = np.select(
        [
            (position > 0) & (previous_position <= 0),
            (position <= 0) & (previous_position > 0),
            position > 0,
        ],
        ["enter", "exit", "hold"],
        default="flat",
    )
    out["trade_change"] = trade_change
    out["net_return"] = net_returns.fillna(0)
    out["account_equity"] = equity
    out["equity_before_bar"] = equity.shift(1).fillna(initial_capital)
    out["benchmark_equity"] = benchmark_equity
    out["drawdown_pct"] = drawdown * 100
    out["benchmark_drawdown_pct"] = benchmark_drawdown * 100
    out["unrealized_position_value"] = np.where(position > 0, equity, 0.0)
    out["cash_estimate"] = np.where(position > 0, 0.0, equity)
    return out


def build_trade_ledger(equity_frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    position = equity_frame["position"].fillna(0).to_numpy()
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

    if not trades:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "status",
                "entry_ts",
                "exit_ts",
                "bars_held",
                "entry_price",
                "exit_price",
                "quantity_estimate",
                "gross_return_pct",
                "net_return_pct",
                "entry_equity",
                "exit_equity",
                "net_pnl",
                "max_adverse_excursion_pct",
                "max_favorable_excursion_pct",
                "exit_reason",
            ]
        )
    return pd.DataFrame(trades)


def _trade_row(
    equity_frame: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    trade_id: int,
    status: str,
) -> dict:
    segment = equity_frame.iloc[entry_idx : exit_idx + 1]
    held_segment = segment[segment["position"] > 0]
    entry = equity_frame.iloc[entry_idx]
    exit_row = equity_frame.iloc[exit_idx]
    entry_equity = float(entry["equity_before_bar"])
    exit_equity = float(exit_row["account_equity"])
    entry_price = float(entry["close"])
    exit_price = float(exit_row["close"])
    gross_return = exit_price / entry_price - 1
    net_return = exit_equity / max(entry_equity, 1e-9) - 1
    quantity = entry_equity / entry_price if entry_price else 0.0
    lows = held_segment["low"] if not held_segment.empty else segment["low"]
    highs = held_segment["high"] if not held_segment.empty else segment["high"]
    exit_reason = "open_at_end" if status == "open" else "strategy_exit_or_risk_rule"

    return {
        "trade_id": trade_id,
        "status": status,
        "entry_ts": entry["ts_open"],
        "exit_ts": exit_row["ts_open"],
        "bars_held": int((held_segment["position"] > 0).sum()) if not held_segment.empty else 0,
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "quantity_estimate": round(quantity, 8),
        "gross_return_pct": round(gross_return * 100, 4),
        "net_return_pct": round(net_return * 100, 4),
        "entry_equity": round(entry_equity, 2),
        "exit_equity": round(exit_equity, 2),
        "net_pnl": round(exit_equity - entry_equity, 2),
        "max_adverse_excursion_pct": round((float(lows.min()) / entry_price - 1) * 100, 4),
        "max_favorable_excursion_pct": round((float(highs.max()) / entry_price - 1) * 100, 4),
        "exit_reason": exit_reason,
    }


def write_svg_line_chart(
    path: Path,
    frame: pd.DataFrame,
    *,
    y_column: str,
    title: str,
    y_label: str,
    stroke: str,
) -> None:
    values = frame[y_column].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    width = 960
    height = 360
    margin_left = 72
    margin_right = 28
    margin_top = 46
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    points = []
    denominator = max(len(values) - 1, 1)
    for idx, value in enumerate(values):
        x = margin_left + (idx / denominator) * plot_width
        y = margin_top + (1 - ((float(value) - y_min) / (y_max - y_min))) * plot_height
        points.append(f"{x:.2f},{y:.2f}")

    start_label = str(frame["ts_open"].iloc[0])[:16] if not frame.empty else ""
    end_label = str(frame["ts_open"].iloc[-1])[:16] if not frame.empty else ""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="26" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>
  <text x="18" y="{height / 2:.0f}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563" transform="rotate(-90 18 {height / 2:.0f})">{html.escape(y_label)}</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#d1d5db"/>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#d1d5db"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{width - margin_right}" y2="{margin_top}" stroke="#f3f4f6"/>
  <text x="{margin_left - 8}" y="{margin_top + 4}" font-family="Arial, sans-serif" font-size="11" fill="#4b5563" text-anchor="end">{y_max:.2f}</text>
  <text x="{margin_left - 8}" y="{height - margin_bottom + 4}" font-family="Arial, sans-serif" font-size="11" fill="#4b5563" text-anchor="end">{y_min:.2f}</text>
  <text x="{margin_left}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{html.escape(start_label)}</text>
  <text x="{width - margin_right}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#4b5563" text-anchor="end">{html.escape(end_label)}</text>
  <polyline fill="none" stroke="{stroke}" stroke-width="2.4" points="{' '.join(points)}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_outputs(
    repo_root: Path,
    equity_frame: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: PerformanceMetrics,
    *,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
    interval_minutes: int,
) -> PaperLedgerResult:
    blueprint = selected_blueprint()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "artifacts" / "paper_trading" / f"{STRATEGY_ID}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_ledger_path = out_dir / "paper_ledger.csv"
    equity_curve_path = out_dir / "equity_curve.csv"
    trades_path = out_dir / "per_trade_ledger.csv"
    equity_chart_path = out_dir / "equity_curve.svg"
    drawdown_chart_path = out_dir / "drawdown_curve.svg"
    summary_path = out_dir / "summary.json"

    equity_frame.to_csv(paper_ledger_path, index=False)
    equity_frame[
        [
            "ts_open",
            "close",
            "raw_signal",
            "position",
            "net_return",
            "account_equity",
            "benchmark_equity",
            "drawdown_pct",
            "benchmark_drawdown_pct",
        ]
    ].to_csv(equity_curve_path, index=False)
    trades.to_csv(trades_path, index=False)
    write_svg_line_chart(
        equity_chart_path,
        equity_frame,
        y_column="account_equity",
        title="SOL Volume Expansion Breakout Paper Equity",
        y_label="Account equity",
        stroke="#0f766e",
    )
    write_svg_line_chart(
        drawdown_chart_path,
        equity_frame,
        y_column="drawdown_pct",
        title="SOL Volume Expansion Breakout Drawdown",
        y_label="Drawdown percent",
        stroke="#b91c1c",
    )

    latest = equity_frame.iloc[-1]
    latest_state = {
        "latest_timestamp": str(latest["ts_open"]),
        "latest_close": float(latest["close"]),
        "latest_raw_signal": int(latest["raw_signal"]),
        "latest_delayed_position": int(latest["position"]),
        "paper_state": "paper_hold" if int(latest["position"]) else "paper_flat",
        "live_trading": "not_used",
    }
    result = PaperLedgerResult(
        strategy_id=STRATEGY_ID,
        strategy_name=blueprint.name,
        created_at=datetime.now(UTC).isoformat(),
        summary_path=str(summary_path),
        paper_ledger_csv=str(paper_ledger_path),
        equity_curve_csv=str(equity_curve_path),
        trades_csv=str(trades_path),
        equity_chart_svg=str(equity_chart_path),
        drawdown_chart_svg=str(drawdown_chart_path),
        metrics=metrics,
        latest_state=latest_state,
    )
    payload = {
        "created_at": result.created_at,
        "strategy": {
            "strategy_id": STRATEGY_ID,
            "name": blueprint.name,
            "asset": blueprint.asset,
            "hypothesis": blueprint.hypothesis,
            "entry_logic": blueprint.entry_logic,
            "exit_logic": blueprint.exit_logic,
        },
        "assumptions": {
            "interval_minutes": interval_minutes,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "execution": "one-bar delayed close-to-close paper simulation",
            "live_trading": "not_used",
        },
        "metrics": asdict(metrics),
        "latest_state": latest_state,
        "trade_count": len(trades),
        "outputs": {
            "paper_ledger_csv": result.paper_ledger_csv,
            "equity_curve_csv": result.equity_curve_csv,
            "trades_csv": result.trades_csv,
            "equity_chart_svg": result.equity_chart_svg,
            "drawdown_chart_svg": result.drawdown_chart_svg,
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def run_paper_ledger(
    repo_root: Path,
    *,
    interval_minutes: int = 60,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 10_000.0,
) -> PaperLedgerResult:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    equity_frame, metrics = build_paper_frame(
        settings,
        interval_minutes=interval_minutes,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )
    trades = build_trade_ledger(equity_frame, initial_capital)
    return write_outputs(
        repo_root,
        equity_frame,
        trades,
        metrics,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        interval_minutes=interval_minutes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the SOL breakout paper-trading ledger.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_paper_ledger(
        args.repo_root.resolve(),
        interval_minutes=args.interval_minutes,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
    )
    print(f"summary={result.summary_path}")
    print(f"paper_ledger={result.paper_ledger_csv}")
    print(f"per_trade_ledger={result.trades_csv}")
    print(f"equity_chart={result.equity_chart_svg}")
    print(f"drawdown_chart={result.drawdown_chart_svg}")
    print(
        f"net_return_pct={result.metrics.net_return_pct:.2f} "
        f"max_drawdown_pct={result.metrics.max_drawdown_pct:.2f} "
        f"trades={result.metrics.trades} "
        f"state={result.latest_state['paper_state']}"
    )


if __name__ == "__main__":
    main()
