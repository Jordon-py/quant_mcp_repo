"""Shared FastMCP server instance.

Tool, resource, and prompt modules import this object for registration. Keeping
one instance prevents split MCP surfaces across transports.
"""

from __future__ import annotations

from fastmcp import FastMCP

from quant_mcp.mcp.workflow_prompts import CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT

mcp = FastMCP("Quant Research MCP", instructions=CORE_STRATEGY_RESEARCH_SYSTEM_PROMPT)
