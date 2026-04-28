"""Read-only MCP resources for operational state.

Resources expose inspectable system data without changing datasets, approvals,
or execution state.
"""

from __future__ import annotations

import json

from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

from quant_mcp.mcp.server import mcp
from quant_mcp.mcp.workflow_prompts import (
    CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT,
    GENERIC_STRATEGY_CRITIQUE_PROMPT,
    ML_RL_STRATEGY_CREATION_PROMPT,
    PROMPT_POLICY,
)
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


@mcp.resource("quant://prompts/strategy-research-workflow", mime_type="text/plain")
async def strategy_research_workflow_resource(ctx: Context = CurrentContext()) -> str:
    """Read-only copy of the core strategy research workflow prompt."""
    await ctx.info("Reading core strategy workflow prompt")
    return CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT


@mcp.resource("quant://prompts/generic-strategy-critique", mime_type="text/plain")
async def generic_strategy_critique_resource(ctx: Context = CurrentContext()) -> str:
    """Read-only copy of the generic strategy critique workflow prompt."""
    await ctx.info("Reading generic strategy critique prompt")
    return GENERIC_STRATEGY_CRITIQUE_PROMPT


@mcp.resource("quant://prompts/ml-rl-strategy-creation", mime_type="text/plain")
async def ml_rl_strategy_creation_resource(ctx: Context = CurrentContext()) -> str:
    """Read-only copy of the institutional ML/RL strategy creation workflow prompt."""
    await ctx.info("Reading ML/RL strategy creation prompt")
    return ML_RL_STRATEGY_CREATION_PROMPT


@mcp.resource("quant://system/workflow-policy", mime_type="application/json")
async def workflow_policy_resource(ctx: Context = CurrentContext()) -> str:
    """Read-only metadata describing how strategy workflow prompts are exposed."""
    await ctx.info("Reading workflow prompt policy")
    return json.dumps(PROMPT_POLICY, indent=2)
