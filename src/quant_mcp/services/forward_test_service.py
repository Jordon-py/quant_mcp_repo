"""Forward-test gate placeholder.

This service preserves the promotion step in the architecture until a persistent
paper-trading ledger replaces the current optimistic placeholder.
"""

from __future__ import annotations

from quant_mcp.domain.validation import ForwardTestRequest, ForwardTestResult
from quant_mcp.enums import ValidationStatus
from quant_mcp.settings import AppSettings


class ForwardTestService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run_forward_test(self, request: ForwardTestRequest) -> ForwardTestResult:
        return ForwardTestResult(
            strategy_id=request.strategy_id,
            dataset_id=request.dataset_id,
            status=ValidationStatus.WARNING,
            paper_path_ready=True,
            notes="Placeholder forward-test gate. Replace with persistent paper ledger before promoting live.",
        )
