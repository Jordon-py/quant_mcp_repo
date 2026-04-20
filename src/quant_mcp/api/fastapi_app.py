from __future__ import annotations

from fastapi import FastAPI

from quant_mcp.mcp import prompts, resources, tools  # noqa: F401  Ensures registration side effects.
from quant_mcp.mcp.server import mcp
from quant_mcp.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.mount(settings.mcp_path, mcp.http_app(path="/"))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "mcp_path": settings.mcp_path}

    return app


app = create_app()
