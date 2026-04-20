from __future__ import annotations

from fastmcp.prompts import Message, PromptResult

from quant_mcp.mcp.server import mcp


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
