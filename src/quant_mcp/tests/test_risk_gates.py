from datetime import timedelta

from quant_mcp.domain.approval import ApprovalRecord
from quant_mcp.domain.common import utc_now
from quant_mcp.domain.execution import TradeIntent
from quant_mcp.enums import ExecutionMode, OrderSide
from quant_mcp.services.risk_service import RiskService
from quant_mcp.settings import AppSettings


def make_settings(**kwargs) -> AppSettings:
    base = AppSettings(enable_live_trading=False)
    return base.model_copy(update=kwargs)


def test_live_trade_rejected_when_environment_disabled():
    settings = make_settings(enable_live_trading=False)
    risk = RiskService(settings)
    intent = TradeIntent(
        strategy_id="s1",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=1,
        execution_mode=ExecutionMode.LIVE,
        requested_allocation_pct=0.01,
        client_order_id="abc",
    )
    result = risk.validate_live_trade(intent, None, strategy_passed_validation=True, paper_path_exists=True)
    assert result.allowed is False
    assert result.code == "live_disabled"


def test_live_trade_rejected_without_approval():
    settings = make_settings(enable_live_trading=True)
    risk = RiskService(settings)
    intent = TradeIntent(
        strategy_id="s1",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=1,
        execution_mode=ExecutionMode.LIVE,
        requested_allocation_pct=0.01,
        client_order_id="abc",
    )
    result = risk.validate_live_trade(intent, None, strategy_passed_validation=True, paper_path_exists=True)
    assert result.allowed is False
    assert result.code == "approval_missing"


def test_live_trade_rejected_when_allocation_exceeds_limits():
    settings = make_settings(enable_live_trading=True, max_live_allocation_pct=0.02)
    risk = RiskService(settings)
    approval = ApprovalRecord(
        strategy_id="s1",
        symbols=["BTC/USD"],
        max_allocation_pct=0.02,
        approved_by="tester",
        expires_at=utc_now() + timedelta(days=1),
    )
    intent = TradeIntent(
        strategy_id="s1",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=1,
        execution_mode=ExecutionMode.LIVE,
        requested_allocation_pct=0.10,
        client_order_id="abc",
    )
    result = risk.validate_live_trade(intent, approval, strategy_passed_validation=True, paper_path_exists=True)
    assert result.allowed is False
    assert result.code == "allocation_exceeded"
