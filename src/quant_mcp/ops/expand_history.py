"""Backfill longer closed-candle Kraken datasets for research.

The daily append job keeps datasets current. This module is the one-time or
occasional companion that paginates Kraken OHLC history so strategy research is
not limited to the default short exchange response.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_mcp.adapters.kraken.public_client import KrakenPublicClient
from quant_mcp.domain.dataset import Candle
from quant_mcp.ops.daily_data_append import validate_dataset_frame
from quant_mcp.services.dataset_service import DatasetService
from quant_mcp.settings import AppSettings


DEFAULT_SYMBOLS = ("BTC/USD", "SOL/USD")
DEFAULT_TARGET_ROWS = 720


@dataclass(frozen=True)
class HistoryExpansionResult:
    symbol: str
    dataset_id: str
    rows_before: int
    rows_after: int
    rows_added: int
    target_rows: int
    first_ts: str | None
    last_ts: str | None
    duplicate_rows: int
    path: str
    status: str
    target_met: bool
    shortfall_rows: int
    notes: str


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def starting_since_unix(target_rows: int, interval_minutes: int) -> int:
    """Start far enough back that tailing `target_rows` still reaches recent data."""
    extra_bars = max(240, int(target_rows * 0.15))
    lookback_minutes = (target_rows + extra_bars) * interval_minutes
    return int((datetime.now(UTC) - timedelta(minutes=lookback_minutes)).timestamp())


async def collect_ohlc_history(
    client: KrakenPublicClient,
    *,
    symbol: str,
    interval_minutes: int,
    target_rows: int,
    since_unix: int | None = None,
    max_pages: int = 12,
) -> list[Candle]:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")

    since = since_unix if since_unix is not None else starting_since_unix(target_rows, interval_minutes)
    candles_by_key: dict[tuple[str, int, datetime], Candle] = {}

    for _ in range(max_pages):
        page, last_cursor = await client.fetch_ohlc_page(symbol, interval_minutes, since)
        if not page:
            break

        for candle in page:
            candles_by_key[(candle.symbol, candle.interval_minutes, candle.ts_open)] = candle

        latest_closed = max((candle.ts_close for candle in page if candle.closed), default=None)
        if latest_closed and latest_closed >= datetime.now(UTC) - timedelta(minutes=interval_minutes * 2):
            break

        next_since = last_cursor or int(max(candle.ts_open.timestamp() for candle in page))
        if next_since <= since:
            break
        since = next_since

    return sorted(candles_by_key.values(), key=lambda candle: candle.ts_open)


async def expand_symbol_history(
    service: DatasetService,
    *,
    symbol: str,
    interval_minutes: int,
    target_rows: int,
    since_unix: int | None = None,
    max_pages: int = 12,
) -> HistoryExpansionResult:
    dataset_id = service.dataset_id(symbol, interval_minutes)
    relative_path = f"datasets/{dataset_id}.parquet"
    path = service.dataset_path(dataset_id)
    existing = service.store.read_frame(relative_path) if path.exists() else pd.DataFrame()
    rows_before = len(existing)

    candles = await collect_ohlc_history(
        service.client,
        symbol=symbol,
        interval_minutes=interval_minutes,
        target_rows=target_rows,
        since_unix=since_unix,
        max_pages=max_pages,
    )
    fetched = pd.DataFrame([candle.model_dump(mode="json") for candle in candles])
    combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
    combined = service._closed_sorted_deduped(combined)
    if combined.empty:
        raise ValueError(f"No closed candles available for {dataset_id}")

    combined = combined.tail(target_rows)
    validate_dataset_frame(combined, dataset_id)
    if len(combined) < rows_before:
        raise ValueError(f"{dataset_id} row count would shrink from {rows_before} to {len(combined)}")

    out_path = service.store.write_frame(relative_path, combined)
    profile = service.profile_dataset(dataset_id)
    target_met = profile.rows >= target_rows
    shortfall = max(target_rows - profile.rows, 0)
    notes = (
        "Target reached with closed Kraken candles."
        if target_met
        else "Kraken public OHLC returned fewer closed rows than requested; use an external historical source for deeper backfill."
    )
    return HistoryExpansionResult(
        symbol=symbol,
        dataset_id=dataset_id,
        rows_before=rows_before,
        rows_after=profile.rows,
        rows_added=profile.rows - rows_before,
        target_rows=target_rows,
        first_ts=profile.first_ts.isoformat() if profile.first_ts else None,
        last_ts=profile.last_ts.isoformat() if profile.last_ts else None,
        duplicate_rows=profile.duplicate_rows,
        path=str(out_path),
        status="ok",
        target_met=target_met,
        shortfall_rows=shortfall,
        notes=notes,
    )


async def run_expansion(
    repo_root: Path,
    *,
    symbols: list[str],
    interval_minutes: int,
    target_rows: int,
    since_unix: int | None = None,
    max_pages: int = 12,
) -> dict:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    service = DatasetService(settings)
    started_at = datetime.now(UTC)
    results = []
    for symbol in symbols:
        result = await expand_symbol_history(
            service,
            symbol=symbol,
            interval_minutes=interval_minutes,
            target_rows=target_rows,
            since_unix=since_unix,
            max_pages=max_pages,
        )
        results.append(result)

    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "symbols": symbols,
        "interval_minutes": interval_minutes,
        "target_rows": target_rows,
        "results": [asdict(result) for result in results],
    }
    write_expansion_report(repo_root, payload)
    return payload


def write_expansion_report(repo_root: Path, payload: dict) -> Path:
    out_dir = repo_root / "artifacts" / "data_backfill"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"history_expansion_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand Kraken OHLC datasets beyond one short page.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--since-unix", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = asyncio.run(
            run_expansion(
                args.repo_root.resolve(),
                symbols=args.symbols,
                interval_minutes=args.interval_minutes,
                target_rows=args.target_rows,
                since_unix=args.since_unix,
                max_pages=args.max_pages,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly for scheduled/research use.
        print(f"history_expansion=fail error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "history_expansion=ok "
        f"symbols={','.join(args.symbols)} "
        f"target_rows={payload['target_rows']}"
    )
    for item in payload["results"]:
        print(
            f"{item['dataset_id']} rows_before={item['rows_before']} "
            f"rows_after={item['rows_after']} last_ts={item['last_ts']}"
        )


if __name__ == "__main__":
    main()
