from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from quant_mcp.domain.common import utc_now
from quant_mcp.enums import ExecutionMode, OrderSide


class RiskRejection(BaseModel):
    allowed: bool = False
    code: str
    reason: str


class RiskApproval(BaseModel):
    allowed: bool = True
    code: str = "ok"
    reason: str = "approved"


class TradeIntent(BaseModel):
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    symbol: str
    side: OrderSide
    quantity: float
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    requested_allocation_pct: float = Field(gt=0)
    client_order_id: str
    created_at: datetime = Field(default_factory=utc_now)


class PaperTradeStepRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    bar_index: int


class OrderPlan(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str = "market"
