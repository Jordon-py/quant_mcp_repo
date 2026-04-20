from __future__ import annotations

from quant_mcp.logging import configure_logging
from quant_mcp.mcp import prompts, resources, tools  # noqa: F401
from quant_mcp.mcp.server import mcp
from quant_mcp.settings import get_settings


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    mcp.run(transport="http", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    run()
