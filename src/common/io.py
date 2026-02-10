"""
Common I/O helpers (baca/tulis CSV, Parquet, config, dsb.).
"""

from pathlib import Path
from typing import Any

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


