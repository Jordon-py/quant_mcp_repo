"""Read-only MCP resources for operational state.

Resources expose inspectable system data without changing datasets, approvals,
or execution state.
"""

from __future__ import annotations

import json

from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

from quant_mcp.mcp.server import mcp
from quant_mcp.services.risk_service import RiskService
from quant_mcp.settings import get_settings


@mcp.resource("quant://system/risk-status", mime_type="application/json")
async def risk_status_resource(ctx: Context = CurrentContext()) -> str:
    """Read-only view of the current global risk configuration."""
    await ctx.info("Reading risk status resource")
    risk = RiskService(get_settings())
    return json.dumps(risk.get_risk_status(), indent=2)


@mcp.resource("quant://datasets/{dataset_id}/profile", mime_type="application/json")
async def dataset_profile_resource(dataset_id: str, ctx: Context = CurrentContext()) -> str:
    """Dynamic dataset profile resource for a persisted dataset."""
    from quant_mcp.services.dataset_service import DatasetService

    service = DatasetService(get_settings())
    profile = service.profile_dataset(dataset_id)
    await ctx.info(f"Read profile for {dataset_id}")
    return profile.model_dump_json(indent=2)
