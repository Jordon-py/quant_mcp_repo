"""MCP tool wrappers for the quant research workflow.

This layer translates MCP calls into typed domain requests and delegates to
services. It intentionally stays thin so trading rules remain testable outside
the protocol boundary.
"""

from __future__ import annotations

from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

from quant_mcp.domain.approval import ApprovalRecord
from quant_mcp.domain.dataset import (
    FeatureTableRequest,
    FeatureTableResult,
    IngestMarketDataRequest,
    RefreshDatasetRequest,
)
from quant_mcp.domain.execution import OrderPlan, PaperTradeStepRequest, RiskApproval, RiskRejection, TradeIntent
from quant_mcp.domain.strategy import GenerateStrategyCandidatesRequest, StrategyListResult, StrategySpec
from quant_mcp.domain.validation import (
    BacktestRequest,
    BacktestResult,
    ForwardTestRequest,
    ForwardTestResult,
    WalkForwardRequest,
    WalkForwardResult,
)
from quant_mcp.mcp.server import mcp
from quant_mcp.services.approval_service import ApprovalService
from quant_mcp.services.backtest_service import BacktestService
from quant_mcp.services.dataset_service import DatasetService
from quant_mcp.services.execution_service import ExecutionService
from quant_mcp.services.feature_service import FeatureService
from quant_mcp.services.forward_test_service import ForwardTestService
from quant_mcp.services.risk_service import RiskService
from quant_mcp.services.strategy_service import StrategyService
from quant_mcp.services.walkforward_service import WalkForwardService
from quant_mcp.settings import get_settings


def _services() -> tuple[
    DatasetService,
    FeatureService,
    StrategyService,
    BacktestService,
    WalkForwardService,
    ForwardTestService,
    ApprovalService,
    RiskService,
    ExecutionService,
]:
    settings = get_settings()
    # Services are lightweight and stateless per call; persisted state lives under configured roots.
    approvals = ApprovalService(settings)
    risk = RiskService(settings)
    return (
        DatasetService(settings),
        FeatureService(settings),
        StrategyService(settings),
        BacktestService(settings),
        WalkForwardService(settings),
        ForwardTestService(settings),
        approvals,
        risk,
        ExecutionService(settings, approvals, risk),
    )


@mcp.tool
async def health_check(ctx: Context = CurrentContext()) -> dict:
    """Return process health, transport information, and global live-trading state."""
    settings = get_settings()
    await ctx.info("health_check called")
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "transport": ctx.transport,
        "live_enabled": settings.enable_live_trading,
    }


@mcp.tool
async def ingest_market_data(request: IngestMarketDataRequest, ctx: Context = CurrentContext()):
    """Ingest OHLC market data from Kraken public endpoints and persist closed candles only."""
    dataset_service, *_ = _services()
    await ctx.info(f"Ingesting market data for {request.symbol}")
    return await dataset_service.ingest_market_data(request)


@mcp.tool
async def refresh_dataset(request: RefreshDatasetRequest, ctx: Context = CurrentContext()):
    """Refresh an existing dataset using append-only dedupe semantics on closed candles."""
    dataset_service, *_ = _services()
    return await dataset_service.refresh_dataset(request)


@mcp.tool
def profile_dataset(dataset_id: str) -> dict:
    """Profile a saved dataset for row count, time span, duplicates, and null distribution."""
    dataset_service, *_ = _services()
    return dataset_service.profile_dataset(dataset_id).model_dump(mode="json")


@mcp.tool
def list_dataset_versions() -> list[dict]:
    """List current dataset artifacts available on disk."""
    dataset_service, *_ = _services()
    return [v.model_dump(mode="json") for v in dataset_service.list_dataset_versions()]


@mcp.tool
def build_feature_table(request: FeatureTableRequest) -> FeatureTableResult:
    """Build a lagged feature table from a dataset without leaking future information."""
    _, feature_service, *_ = _services()
    return feature_service.build_feature_table(request)


@mcp.tool
def generate_strategy_candidates(request: GenerateStrategyCandidatesRequest) -> list[StrategySpec]:
    """Generate deterministic strategy candidates from a constrained grammar."""
    _, _, strategy_service, *_ = _services()
    return strategy_service.generate_strategy_candidates(request)


@mcp.tool
def save_strategy(spec: StrategySpec) -> dict:
    """Persist a strategy specification to the strategy registry."""
    _, _, strategy_service, *_ = _services()
    path = strategy_service.save_strategy(spec)
    return {"saved": True, "path": str(path), "strategy_id": spec.strategy_id}


@mcp.tool
def list_strategies() -> StrategyListResult:
    """Return all saved strategies."""
    _, _, strategy_service, *_ = _services()
    return strategy_service.list_strategies()


@mcp.tool
def run_backtest(request: BacktestRequest) -> BacktestResult:
    """Run a chronological, fee-aware backtest on a saved feature table."""
    _, _, _, backtest_service, *_ = _services()
    return backtest_service.run_backtest(request)


@mcp.tool
def compare_backtests() -> list[BacktestResult]:
    """Return historical backtest results for comparison."""
    _, _, _, backtest_service, *_ = _services()
    return backtest_service.compare_backtests()


@mcp.tool
def run_walk_forward(request: WalkForwardRequest) -> WalkForwardResult:
    """Run chronological walk-forward validation. Random shuffle is intentionally unsupported."""
    _, _, _, _, walk_service, *_ = _services()
    return walk_service.run_walk_forward(request)


@mcp.tool
def run_forward_test(request: ForwardTestRequest) -> ForwardTestResult:
    """Run the forward-test gate placeholder used before paper/live promotion."""
    _, _, _, _, _, forward_service, *_ = _services()
    return forward_service.run_forward_test(request)


@mcp.tool
def approve_strategy(approval: ApprovalRecord) -> dict:
    """Create an approval record required before any live-trade intent can become executable."""
    *_, approval_service, _, _ = _services()
    path = approval_service.approve_strategy(approval)
    return {"approved": True, "path": str(path), "approval_id": approval.approval_id}


@mcp.tool
def revoke_approval(approval_id: str) -> ApprovalRecord:
    """Revoke a previously-issued approval record."""
    *_, approval_service, _, _ = _services()
    return approval_service.revoke_approval(approval_id)


@mcp.tool
def get_risk_status() -> dict:
    """Return the active global risk configuration."""
    *_, risk_service, _ = _services()
    return risk_service.get_risk_status()


@mcp.tool
def paper_trade_step(request: PaperTradeStepRequest) -> dict:
    """Advance the paper-trading simulation by one logical step."""
    *_, execution_service = _services()
    return execution_service.paper_trade_step(request)


@mcp.tool
def prepare_live_trade_intent(intent: TradeIntent) -> RiskApproval | RiskRejection:
    """Validate whether a live trade intent is eligible. This tool never places an order."""
    *_, execution_service = _services()
    # The v1 MCP surface is intentionally conservative until validation/paper state is wired in.
    return execution_service.prepare_live_trade_intent(
        intent=intent,
        strategy_passed_validation=False,
        paper_path_exists=True,
    )


@mcp.tool
async def execute_live_trade(order: OrderPlan, ctx: Context = CurrentContext()) -> dict:
    """Place a live Kraken order. This should only be called after a successful live-intent preflight."""
    *_, execution_service = _services()
    await ctx.warning("execute_live_trade called; caller is responsible for preflight discipline")
    return await execution_service.execute_live_trade(order)


@mcp.tool
async def cancel_all_live_orders() -> dict:
    """Cancel all open live Kraken orders through the private adapter."""
    *_, execution_service = _services()
    return await execution_service.cancel_all_live_orders()
