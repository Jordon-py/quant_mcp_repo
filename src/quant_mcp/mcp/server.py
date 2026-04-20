"""Shared FastMCP server instance.

Tool, resource, and prompt modules import this object for registration. Keeping
one instance prevents split MCP surfaces across transports.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("Quant Research MCP")
