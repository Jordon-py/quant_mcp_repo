"""Paper/live execution coordinator.

ExecutionService coordinates approvals, risk checks, and the Kraken private
adapter. It does not own strategy selection or validation policy.
"""

from __future__ import annotations

from quant_mcp.adapters.kraken.private_client import KrakenPrivateClient
from quant_mcp.domain.execution import OrderPlan, PaperTradeStepRequest, RiskApproval, RiskRejection, TradeIntent
from quant_mcp.services.approval_service import ApprovalService
from quant_mcp.services.risk_service import RiskService
from quant_mcp.settings import AppSettings


class ExecutionService:
    def __init__(self, settings: AppSettings, approval_service: ApprovalService, risk_service: RiskService) -> None:
        self.settings = settings
        self.approval_service = approval_service
        self.risk_service = risk_service
        # Private trading stays unavailable unless both credentials are present.
        self.private_client = (
            KrakenPrivateClient(
                settings.kraken_api_key,
                settings.kraken_api_secret,
                base_url=settings.kraken_rest_base_url,
            )
            if settings.kraken_api_key and settings.kraken_api_secret
            else None
        )

    def paper_trade_step(self, request: PaperTradeStepRequest) -> dict:
        return {
            "strategy_id": request.strategy_id,
            "dataset_id": request.dataset_id,
            "bar_index": request.bar_index,
            "status": "simulated",
        }

    def prepare_live_trade_intent(
        self,
        intent: TradeIntent,
        strategy_passed_validation: bool,
        paper_path_exists: bool,
    ) -> RiskApproval | RiskRejection:
        # This is a preflight only; placing an order remains a separate privileged call.
        approval = self.approval_service.get_active_approval(intent.strategy_id)
        return self.risk_service.validate_live_trade(intent, approval, strategy_passed_validation, paper_path_exists)

    async def execute_live_trade(self, order: OrderPlan) -> dict:
        if self.private_client is None:
            raise RuntimeError("Kraken private client unavailable; credentials missing")
        payload = {
            "pair": order.symbol,
            "type": order.side.value,
            "ordertype": order.order_type,
            "volume": str(order.quantity),
        }
        return await self.private_client.add_order(payload)

    async def cancel_all_live_orders(self) -> dict:
        if self.private_client is None:
            raise RuntimeError("Kraken private client unavailable; credentials missing")
        return await self.private_client.cancel_all()
