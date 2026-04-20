"""Approval contracts required before live-trading preflight can pass."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, Field

from quant_mcp.domain.common import utc_now
from quant_mcp.enums import ApprovalStatus


class ApprovalRecord(BaseModel):
    approval_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    status: ApprovalStatus = ApprovalStatus.ACTIVE
    symbols: list[str] = Field(default_factory=list)
    max_allocation_pct: float
    approved_by: str
    approved_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(days=30))
    notes: str | None = None

    def is_valid_for_symbol(self, symbol: str, now: datetime | None = None) -> bool:
        current = now or utc_now()
        return (
            self.status == ApprovalStatus.ACTIVE
            and current < self.expires_at
            and symbol in self.symbols
        )
