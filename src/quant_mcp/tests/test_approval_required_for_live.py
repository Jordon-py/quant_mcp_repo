from quant_mcp.domain.execution import TradeIntent
from quant_mcp.enums import ExecutionMode, OrderSide
from quant_mcp.services.approval_service import ApprovalService
from quant_mcp.services.execution_service import ExecutionService
from quant_mcp.services.risk_service import RiskService
from quant_mcp.settings import AppSettings


def test_prepare_live_trade_requires_approval(tmp_path):
    settings = AppSettings(enable_live_trading=True, artifact_dir=tmp_path / "artifacts")
    approvals = ApprovalService(settings)
    risk = RiskService(settings)
    execution = ExecutionService(settings, approvals, risk)
    intent = TradeIntent(
        strategy_id="s1",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=1,
        execution_mode=ExecutionMode.LIVE,
        requested_allocation_pct=0.01,
        client_order_id="c1",
    )
    result = execution.prepare_live_trade_intent(intent, strategy_passed_validation=True, paper_path_exists=True)
    assert result.allowed is False
    assert result.code == "approval_missing"
