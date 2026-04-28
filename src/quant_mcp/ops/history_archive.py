"""Maintain a long-running local Kraken candle archive.

Kraken public OHLC only exposes a recent window on demand. This module turns
future scheduled runs into a growing local archive and also supports importing a
trusted external CSV when deeper backfill data is available.
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

from quant_mcp.domain.dataset import Candle
from quant_mcp.ops.daily_data_append import validate_dataset_frame
from quant_mcp.services.dataset_service import DatasetService
from quant_mcp.settings import AppSettings


DEFAULT_SYMBOLS = ("BTC/USD", "SOL/USD")


@dataclass(frozen=True)
class ArchiveUpdateResult:
    symbol: str
    dataset_id: str
    archive_rows_before: int
    archive_rows_after: int
    archive_rows_added: int
    dataset_rows_after: int
    first_ts: str | None
    last_ts: str | None
    duplicate_rows: int
    archive_path: str
    dataset_path: str | None
    status: str
    notes: str


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def archive_relative_path(dataset_id: str) -> str:
    return f"archives/kraken/{dataset_id}.parquet"


def dataset_relative_path(dataset_id: str) -> str:
    return f"datasets/{dataset_id}.parquet"


def read_frame_if_exists(service: DatasetService, relative_path: str) -> pd.DataFrame:
    path = service.settings.data_dir / relative_path
    if not path.exists():
        return pd.DataFrame()
    return service.store.read_frame(relative_path)


def canonicalize_ohlc_frame(frame: pd.DataFrame, service: DatasetService) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("ts_open", "ts_close"):
        out[column] = pd.to_datetime(out[column], utc=True)
    return service._closed_sorted_deduped(out)


async def update_symbol_archive(
    service: DatasetService,
    *,
    symbol: str,
    interval_minutes: int,
    mirror_dataset: bool = True,
) -> ArchiveUpdateResult:
    dataset_id = service.dataset_id(symbol, interval_minutes)
    archive_rel = archive_relative_path(dataset_id)
    dataset_rel = dataset_relative_path(dataset_id)
    archive = read_frame_if_exists(service, archive_rel)
    dataset = read_frame_if_exists(service, dataset_rel)
    archive_rows_before = len(archive)

    seed = pd.concat([archive, dataset], ignore_index=True) if not dataset.empty else archive
    seed = canonicalize_ohlc_frame(seed, service)
    since_unix = None
    if not seed.empty:
        latest_open = pd.to_datetime(seed["ts_open"].max(), utc=True)
        # Overlap one known candle so exchange corrections are deduped without gaps.
        since_unix = int(latest_open.timestamp())

    new_candles = await service.client.fetch_ohlc(symbol, interval_minutes, since_unix)
    new_frame = pd.DataFrame([candle.model_dump(mode="json") for candle in new_candles])
    combined = pd.concat([seed, new_frame], ignore_index=True) if not seed.empty else new_frame
    combined = canonicalize_ohlc_frame(combined, service)
    if combined.empty:
        raise ValueError(f"No closed candles available for archive {dataset_id}")

    validate_dataset_frame(combined, dataset_id)
    archive_path = service.store.write_frame(archive_rel, combined)
    dataset_path = None
    if mirror_dataset:
        dataset_path = service.store.write_frame(dataset_rel, combined)

    duplicate_rows = int(combined.duplicated(subset=["symbol", "interval_minutes", "ts_open"]).sum())
    notes = (
        "Archive seeded from current dataset and refreshed from Kraken public OHLC. "
        "It will grow only as future closed candles arrive unless an external history import is used."
    )
    return ArchiveUpdateResult(
        symbol=symbol,
        dataset_id=dataset_id,
        archive_rows_before=archive_rows_before,
        archive_rows_after=len(combined),
        archive_rows_added=len(combined) - archive_rows_before,
        dataset_rows_after=len(combined) if mirror_dataset else len(dataset),
        first_ts=pd.to_datetime(combined["ts_open"].min(), utc=True).isoformat(),
        last_ts=pd.to_datetime(combined["ts_open"].max(), utc=True).isoformat(),
        duplicate_rows=duplicate_rows,
        archive_path=str(archive_path),
        dataset_path=str(dataset_path) if dataset_path else None,
        status="ok",
        notes=notes,
    )


def normalize_external_csv(path: Path, *, symbol: str, interval_minutes: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = {column.lower().strip(): column for column in raw.columns}
    timestamp_column = columns.get("ts_open") or columns.get("timestamp") or columns.get("time")
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in columns]
    if timestamp_column is None:
        missing.append("ts_open|timestamp|time")
    if missing:
        raise ValueError(f"External CSV missing required columns: {missing}")

    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "interval_minutes": interval_minutes,
            "ts_open": pd.to_datetime(raw[timestamp_column], utc=True),
            "open": raw[columns["open"]].astype(float),
            "high": raw[columns["high"]].astype(float),
            "low": raw[columns["low"]].astype(float),
            "close": raw[columns["close"]].astype(float),
            "volume": raw[columns["volume"]].astype(float),
            "venue": "external",
        }
    )
    if "ts_close" in columns:
        frame["ts_close"] = pd.to_datetime(raw[columns["ts_close"]], utc=True)
    else:
        frame["ts_close"] = frame["ts_open"] + pd.to_timedelta(interval_minutes, unit="m")
    frame["closed"] = frame["ts_close"] <= datetime.now(UTC) - timedelta(seconds=30)
    return frame


def import_external_history(
    service: DatasetService,
    *,
    path: Path,
    symbol: str,
    interval_minutes: int,
    mirror_dataset: bool = True,
) -> ArchiveUpdateResult:
    dataset_id = service.dataset_id(symbol, interval_minutes)
    archive_rel = archive_relative_path(dataset_id)
    dataset_rel = dataset_relative_path(dataset_id)
    archive = read_frame_if_exists(service, archive_rel)
    dataset = read_frame_if_exists(service, dataset_rel)
    archive_rows_before = len(archive)
    external = normalize_external_csv(path, symbol=symbol, interval_minutes=interval_minutes)
    combined = pd.concat([archive, dataset, external], ignore_index=True)
    combined = canonicalize_ohlc_frame(combined, service)
    if combined.empty:
        raise ValueError(f"External import produced no closed candles for {dataset_id}")

    validate_dataset_frame(combined, dataset_id)
    archive_path = service.store.write_frame(archive_rel, combined)
    dataset_path = None
    if mirror_dataset:
        dataset_path = service.store.write_frame(dataset_rel, combined)

    duplicate_rows = int(combined.duplicated(subset=["symbol", "interval_minutes", "ts_open"]).sum())
    return ArchiveUpdateResult(
        symbol=symbol,
        dataset_id=dataset_id,
        archive_rows_before=archive_rows_before,
        archive_rows_after=len(combined),
        archive_rows_added=len(combined) - archive_rows_before,
        dataset_rows_after=len(combined) if mirror_dataset else len(dataset),
        first_ts=pd.to_datetime(combined["ts_open"].min(), utc=True).isoformat(),
        last_ts=pd.to_datetime(combined["ts_open"].max(), utc=True).isoformat(),
        duplicate_rows=duplicate_rows,
        archive_path=str(archive_path),
        dataset_path=str(dataset_path) if dataset_path else None,
        status="ok",
        notes=f"Imported external history from {path}.",
    )


async def run_archive_update(
    repo_root: Path,
    *,
    symbols: list[str],
    interval_minutes: int,
    mirror_dataset: bool = True,
) -> dict:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    service = DatasetService(settings)
    started_at = datetime.now(UTC)
    results = [
        await update_symbol_archive(
            service,
            symbol=symbol,
            interval_minutes=interval_minutes,
            mirror_dataset=mirror_dataset,
        )
        for symbol in symbols
    ]
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "mode": "kraken_update",
        "symbols": symbols,
        "interval_minutes": interval_minutes,
        "mirror_dataset": mirror_dataset,
        "results": [asdict(result) for result in results],
    }
    write_archive_outputs(repo_root, payload)
    return payload


def run_external_import(
    repo_root: Path,
    *,
    path: Path,
    symbol: str,
    interval_minutes: int,
    mirror_dataset: bool = True,
) -> dict:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    service = DatasetService(settings)
    started_at = datetime.now(UTC)
    result = import_external_history(
        service,
        path=path,
        symbol=symbol,
        interval_minutes=interval_minutes,
        mirror_dataset=mirror_dataset,
    )
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": "ok",
        "mode": "external_csv_import",
        "symbols": [symbol],
        "interval_minutes": interval_minutes,
        "mirror_dataset": mirror_dataset,
        "results": [asdict(result)],
    }
    write_archive_outputs(repo_root, payload)
    return payload


def write_archive_outputs(repo_root: Path, payload: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = repo_root / "artifacts" / "history_archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"history_archive_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "history_archive_health.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain long-running local Kraken candle archives.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--no-mirror-dataset", action="store_true")
    parser.add_argument("--import-csv", type=Path, default=None)
    parser.add_argument("--import-symbol", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mirror_dataset = not args.no_mirror_dataset
    try:
        if args.import_csv:
            if not args.import_symbol:
                raise ValueError("--import-symbol is required when --import-csv is used")
            payload = run_external_import(
                args.repo_root.resolve(),
                path=args.import_csv.resolve(),
                symbol=args.import_symbol,
                interval_minutes=args.interval_minutes,
                mirror_dataset=mirror_dataset,
            )
        else:
            payload = asyncio.run(
                run_archive_update(
                    args.repo_root.resolve(),
                    symbols=args.symbols,
                    interval_minutes=args.interval_minutes,
                    mirror_dataset=mirror_dataset,
                )
            )
    except Exception as exc:  # noqa: BLE001 - CLI should provide one clear failure line.
        print(f"history_archive=fail error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"history_archive={payload['status']} mode={payload['mode']}")
    for item in payload["results"]:
        print(
            f"{item['dataset_id']} archive_rows_before={item['archive_rows_before']} "
            f"archive_rows_after={item['archive_rows_after']} "
            f"last_ts={item['last_ts']}"
        )


if __name__ == "__main__":
    main()
