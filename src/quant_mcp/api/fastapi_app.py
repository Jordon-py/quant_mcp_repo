"""Optional FastAPI wrapper for serving the same MCP server over HTTP.

Use this when a browser, webhook, or HTTP-capable MCP client needs the server.
Codex global config should normally use the stdio entrypoint in quant_mcp.main.
"""

from __future__ import annotations

from fastapi import FastAPI

from quant_mcp.mcp import prompts, resources, tools  # noqa: F401  Ensures registration side effects.
from quant_mcp.mcp.server import mcp
from quant_mcp.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    # FastAPI owns only the HTTP mount; the FastMCP instance remains the source of truth.
    app.mount(settings.mcp_path, mcp.http_app(path="/"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "mcp_path": settings.mcp_path}

    return app


app = create_app()
