"""Runtime entrypoint for the Quant Research MCP server.

This module owns process-level transport selection only. MCP registration is
import-driven, and trading/research behavior stays in the service layer.
"""

from __future__ import annotations

import os

from quant_mcp.logging import configure_logging
from quant_mcp.mcp import prompts, resources, tools  # noqa: F401
from quant_mcp.mcp.server import mcp
from quant_mcp.settings import get_settings


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    transport = os.getenv("MCP_TRANSPORT", "http").strip().lower()

    if transport == "stdio":
        # Stdio MCP clients expect protocol frames only; keep FastMCP's banner off this transport.
        mcp.run(transport="stdio", show_banner=True)
        return

    if transport in {"http", "streamable-http", "streamable_http"}:
        mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)
        return

    raise ValueError(f"Unsupported MCP_TRANSPORT={transport!r}; use 'stdio' or 'http'.")


if __name__ == "__main__":
    run()
