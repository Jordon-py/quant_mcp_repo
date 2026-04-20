from __future__ import annotations

from quant_mcp.domain.approval import ApprovalRecord
from quant_mcp.domain.execution import ExecutionMode, RiskApproval, RiskRejection, TradeIntent
from quant_mcp.settings import AppSettings


class RiskService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def get_risk_status(self) -> dict:
        return {
            "live_enabled": self.settings.enable_live_trading,
            "allowed_live_symbols": self.settings.allowed_live_symbols,
            "max_live_allocation_pct": self.settings.max_live_allocation_pct,
            "max_single_trade_risk_pct": self.settings.max_single_trade_risk_pct,
        }

    def validate_live_trade(
        self,
        intent: TradeIntent,
        approval: ApprovalRecord | None,
        strategy_passed_validation: bool,
        paper_path_exists: bool,
    ) -> RiskApproval | RiskRejection:
        if intent.execution_mode != ExecutionMode.LIVE:
            return RiskApproval(reason="paper intent is always non-live")
        if not self.settings.enable_live_trading:
            return RiskRejection(code="live_disabled", reason="ENABLE_LIVE_TRADING=false")
        if not paper_path_exists:
            return RiskRejection(code="paper_missing", reason="paper-trading path not established")
        if not strategy_passed_validation:
            return RiskRejection(code="validation_missing", reason="strategy failed or skipped validation gates")
        if approval is None:
            return RiskRejection(code="approval_missing", reason="no active approval record")
        if intent.symbol not in self.settings.allowed_live_symbols:
            return RiskRejection(code="symbol_not_allowed", reason="symbol blocked by environment allowlist")
        if not approval.is_valid_for_symbol(intent.symbol):
            return RiskRejection(code="approval_invalid", reason="approval expired, revoked, or symbol not allowed")
        if intent.requested_allocation_pct > min(approval.max_allocation_pct, self.settings.max_live_allocation_pct):
            return RiskRejection(code="allocation_exceeded", reason="requested allocation exceeds risk limits")
        return RiskApproval()
