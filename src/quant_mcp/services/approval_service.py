"""Approval registry service for controlled live-trading eligibility."""

from __future__ import annotations

from pathlib import Path

from quant_mcp.adapters.persistence.json_store import JsonStore
from quant_mcp.domain.approval import ApprovalRecord
from quant_mcp.enums import ApprovalStatus
from quant_mcp.settings import AppSettings


class ApprovalService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.store = JsonStore(settings.artifact_dir)
        self.dir = settings.artifact_dir / "approvals"
        self.dir.mkdir(parents=True, exist_ok=True)

    def approve_strategy(self, approval: ApprovalRecord) -> Path:
        return self.store.write_model(f"approvals/{approval.approval_id}.json", approval)

    def revoke_approval(self, approval_id: str) -> ApprovalRecord:
        approval = self.store.read_model(f"approvals/{approval_id}.json", ApprovalRecord)
        approval.status = ApprovalStatus.REVOKED
        self.store.write_model(f"approvals/{approval_id}.json", approval)
        return approval

    def get_active_approval(self, strategy_id: str) -> ApprovalRecord | None:
        for path in self.dir.glob("*.json"):
            approval = self.store.read_model(f"approvals/{path.name}", ApprovalRecord)
            if approval.strategy_id == strategy_id and approval.status == ApprovalStatus.ACTIVE:
                return approval
        return None
