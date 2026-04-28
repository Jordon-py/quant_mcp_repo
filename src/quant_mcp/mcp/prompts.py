"""MCP prompts that guide research review without bypassing deterministic gates."""

from __future__ import annotations

from fastmcp.prompts import Message, PromptResult

from quant_mcp.mcp.server import mcp
from quant_mcp.mcp.workflow_prompts import (
    CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT,
    GENERIC_STRATEGY_CRITIQUE_PROMPT,
    ML_RL_STRATEGY_CREATION_PROMPT,
)


def _append_strategy_text(prompt: str, strategy_text: str | None) -> str:
    if not strategy_text or not strategy_text.strip():
        return prompt
    return f"{prompt}\n\nStrategy supplied for analysis:\n{strategy_text.strip()}"


@mcp.prompt
async def research_review_prompt(strategy_id: str, dataset_id: str) -> PromptResult:
    """Guide an LLM to critique a strategy using research-first reasoning without making live trading decisions."""
    return PromptResult(
        messages=[
            Message(
                f"Review strategy {strategy_id} against dataset {dataset_id}. Focus on regime robustness, fees, slippage, and failure modes. Do not propose live execution unless explicit validation evidence is cited."
            ),
            Message(
                "I will produce a structured critique covering edge plausibility, leakage checks, transaction cost realism, and do-not-deploy conditions.",
                role="assistant",
            ),
        ],
        description="Research-first quant review prompt",
        meta={"safe_mode": True},
    )


@mcp.prompt
async def strategy_research_workflow_prompt(strategy_text: str | None = None) -> PromptResult:
    """Return the strict strategy intake, critique, backtest, and walk-forward workflow."""
    return PromptResult(
        messages=[Message(_append_strategy_text(CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT, strategy_text))],
        description="Core strategy research and validation workflow",
        meta={"safe_mode": True, "workflow": "core_strategy_research"},
    )


@mcp.prompt
async def generic_strategy_critique_prompt(strategy_text: str | None = None) -> PromptResult:
    """Return the generic fallback strategy critique/backtest/walk-forward workflow."""
    return PromptResult(
        messages=[Message(_append_strategy_text(GENERIC_STRATEGY_CRITIQUE_PROMPT, strategy_text))],
        description="Generic strategy critique, backtest, and walk-forward workflow",
        meta={"safe_mode": True, "workflow": "generic_strategy_critique"},
    )


@mcp.prompt
async def ml_rl_strategy_creation_prompt(strategy_text: str | None = None) -> PromptResult:
    """Return the institutional ML/RL strategy creation and validation workflow."""
    return PromptResult(
        messages=[Message(_append_strategy_text(ML_RL_STRATEGY_CREATION_PROMPT, strategy_text))],
        description="Institutional-grade ML/RL strategy research, backtest, and walk-forward workflow",
        meta={"safe_mode": True, "workflow": "ml_rl_strategy_creation"},
    )
