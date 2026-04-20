"""JSON artifact persistence helpers.

This adapter owns file layout mechanics for Pydantic artifacts while services
own the meaning of each artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


class JsonStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_model(self, relative_path: str, model: BaseModel) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_json(self, relative_path: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def read_model(self, relative_path: str, model_type: type[T]) -> T:
        path = self.root / relative_path
        data = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(model_type).validate_python(data)
