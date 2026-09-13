"""Tiny helper for writing the per-stage CSV inventories."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class InventoryWriter:
    """Accumulate rows in memory, flush to a ``;``-separated CSV."""

    def __init__(self, path, columns: list[str] | None = None):
        self.path = Path(path)
        self.columns = columns
        self._rows: list[dict] = []

    def add(self, **row) -> None:
        self._rows.append(row)

    def extend(self, rows) -> None:
        self._rows.extend(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows)
        if self.columns:
            for c in self.columns:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[self.columns]
        return df

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(self.path, index=False, sep=";")
        return self.path


def read_inventory(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")
