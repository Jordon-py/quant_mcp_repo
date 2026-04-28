"""DataFrame persistence adapter for datasets and feature tables.

Parquet is the preferred storage format; pickle is a local-development fallback
so the teaching repo still runs before optional parquet engines are installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd


class ParquetStore:
    """Persist DataFrames with parquet.

    Uses a temporary file and atomic replace to prevent partial writes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_frame(self, relative_path: str, frame: pd.DataFrame) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
        return path

    def read_frame(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        return pd.read_parquet(path)
