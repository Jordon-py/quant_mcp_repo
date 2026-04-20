"""DataFrame persistence adapter for datasets and feature tables.

Parquet is the preferred storage format; pickle is a local-development fallback
so the teaching repo still runs before optional parquet engines are installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    """Persist DataFrames with parquet when available and pickle as a local fallback.

    The fallback keeps the starter repo runnable in minimal environments where pyarrow
    is not yet installed, while still preserving the same logical persistence boundary.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_frame(self, relative_path: str, frame: pd.DataFrame) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            frame.to_pickle(path)
        return path

    def read_frame(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)
