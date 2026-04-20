"""Shared domain primitives.

Common helpers stay dependency-light so all Pydantic contracts can import them
without pulling in services, adapters, or protocol code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactRef(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: str
    path: str
    created_at: datetime = Field(default_factory=utc_now)
    meta: dict[str, Any] = Field(default_factory=dict)
