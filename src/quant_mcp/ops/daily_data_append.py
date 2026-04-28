"""Append closed Kraken candles for BTC/SOL research datasets.

This job is safe to schedule because it uses public market data only, keeps
live-trading disabled, validates row monotonicity, and logs every run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_mcp.domain.dataset import IngestMarketDataRequest, RefreshDatasetRequest
from quant_mcp.services.dataset_service import DatasetService
from quant_mcp.settings import AppSettings


DEFAULT_SYMBOLS = ("BTC/USD", "SOL/USD")


@dataclass(frozen=True)
class SymbolUpdateResult:
    symbol: str
    dataset_id: str
    action: str
    rows_before: int
    rows_after: int
    rows_added: int
    first_ts: str | None
    last_ts: str | None
    duplicate_rows: int
    status: str


class LockFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "LockFile":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode("utf-8"))
        except FileExistsError as exc:
            raise RuntimeError(f"Daily data append already running: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def setup_logging(repo_root: Path) -> logging.Logger:
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quant_mcp.daily_data_append")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "daily_data_append.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def row_count(service: DatasetService, dataset_id: str) -> int:
    path = service.dataset_path(dataset_id)
    if not path.exists():
        return 0
    return len(service.store.read_frame(f"datasets/{dataset_id}.parquet"))


def validate_dataset_frame(frame: pd.DataFrame, dataset_id: str) -> None:
    required = {"symbol", "interval_minutes", "ts_open", "ts_close", "open", "high", "low", "close", "volume", "closed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_id} missing required columns: {sorted(missing)}")
    duplicate_rows = frame.duplicated(subset=["symbol", "interval_minutes", "ts_open"]).sum()
    if duplicate_rows:
        raise ValueError(f"{dataset_id} has {int(duplicate_rows)} duplicate timestamp rows")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{dataset_id} contains nulls in required columns")
    if not frame["closed"].astype(bool).all():
        raise ValueError(f"{dataset_id} contains rows marked not closed")
    ts_close = pd.to_datetime(frame["ts_close"], utc=True)
    if ts_close.max() > datetime.now(UTC) - timedelta(seconds=30):
        raise ValueError(f"{dataset_id} contains an unfinished candle")
    if not pd.to_datetime(frame["ts_open"], utc=True).is_monotonic_increasing:
        raise ValueError(f"{dataset_id} timestamps are not sorted")


async def update_symbol(service: DatasetService, symbol: str, interval_minutes: int, logger: logging.Logger) -> SymbolUpdateResult:
    dataset_id = service.dataset_id(symbol, interval_minutes)
    before = row_count(service, dataset_id)
    if before == 0:
        logger.info("Bootstrapping %s %sm dataset", symbol, interval_minutes)
        action = "ingest"
        await service.ingest_market_data(
            IngestMarketDataRequest(symbol=symbol, interval_minutes=interval_minutes, max_rows=0)
        )
    else:
        logger.info("Refreshing %s %sm dataset from %s existing rows", symbol, interval_minutes, before)
        action = "refresh"
        await service.refresh_dataset(
            RefreshDatasetRequest(symbol=symbol, interval_minutes=interval_minutes, max_rows=0)
        )

    after_frame = service.store.read_frame(f"datasets/{dataset_id}.parquet")
    validate_dataset_frame(after_frame, dataset_id)
    after = len(after_frame)
    if after < before:
        raise ValueError(f"{dataset_id} row count shrank from {before} to {after}")
    profile = service.profile_dataset(dataset_id)
    return SymbolUpdateResult(
        symbol=symbol,
        dataset_id=dataset_id,
        action=action,
        rows_before=before,
        rows_after=after,
        rows_added=after - before,
        first_ts=profile.first_ts.isoformat() if profile.first_ts else None,
        last_ts=profile.last_ts.isoformat() if profile.last_ts else None,
        duplicate_rows=profile.duplicate_rows,
        status="ok",
    )


async def run_update(repo_root: Path, symbols: list[str], interval_minutes: int) -> dict:
    settings = AppSettings(data_dir=repo_root / "data", artifact_dir=repo_root / "artifacts")
    logger = setup_logging(repo_root)
    service = DatasetService(settings)
    started_at = datetime.now(UTC)
    results: list[SymbolUpdateResult] = []
    errors: list[str] = []

    with LockFile(repo_root / "logs" / "daily_data_append.lock"):
        for symbol in symbols:
            try:
                result = await update_symbol(service, symbol, interval_minutes, logger)
                logger.info(
                    "%s %s rows_before=%s rows_after=%s rows_added=%s",
                    result.status,
                    result.dataset_id,
                    result.rows_before,
                    result.rows_after,
                    result.rows_added,
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - scheduled job should log and continue to summarize.
                message = f"{symbol}: {exc}"
                logger.exception("Failed to update %s", symbol)
                errors.append(message)

    status = "ok" if not errors else "fail"
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "status": status,
        "symbols": symbols,
        "interval_minutes": interval_minutes,
        "results": [asdict(result) for result in results],
        "errors": errors,
    }
    write_run_outputs(repo_root, payload)
    if errors:
        raise RuntimeError("; ".join(errors))
    return payload


def write_run_outputs(repo_root: Path, payload: dict) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = repo_root / "artifacts" / "data_append"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"daily_data_append_{stamp}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "daily_data_health.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append closed Kraken candles for BTC/SOL datasets.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--interval-minutes", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = asyncio.run(run_update(args.repo_root.resolve(), args.symbols, args.interval_minutes))
    except Exception as exc:  # noqa: BLE001 - CLI must produce a clear nonzero failure.
        print(f"daily_data_append=fail error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"daily_data_append={payload['status']} symbols={','.join(args.symbols)}")


if __name__ == "__main__":
    main()
