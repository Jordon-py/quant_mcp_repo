"""CLI entrypoint for the critique-driven BTC/SOL strategy expansion report."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_mcp.research.strategy_expansion import run_research


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the critique-driven strategy expansion report.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_module())
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_research(
        args.repo_root.resolve(),
        interval_minutes=args.interval_minutes,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    ranked = sorted(result.evaluations, key=lambda item: item.robustness_score, reverse=True)
    print(f"markdown_report={result.markdown_path}")
    print(f"json_report={result.json_path}")
    print("rank asset strategy score net% sharpe max_dd% wf_pos%")
    for rank, item in enumerate(ranked, 1):
        print(
            f"{rank} {item.blueprint.asset} {item.blueprint.strategy_id} "
            f"{item.robustness_score:.2f} {item.full_period.net_return_pct:.2f} "
            f"{item.full_period.sharpe:.2f} {item.full_period.max_drawdown_pct:.2f} "
            f"{item.walk_forward_positive_fold_rate_pct:.2f}"
        )


if __name__ == "__main__":
    main()
