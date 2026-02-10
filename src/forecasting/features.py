"""
Feature engineering utilities for demand forecasting.

Catatan:
- Saat ini pipeline asli masih berada di `etl/build_features.py`.
- Modul ini disiapkan sebagai rumah baru untuk logika feature engineering
  (mis. baca raw data, agregasi mingguan, lag/rolling features, dsb).
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_processed_features(path: Path = Path("data/processed/weekly_features.parquet")) -> Optional[pd.DataFrame]:
    """
    Helper sederhana untuk membaca fitur terproses jika sudah ada.
    """
    if not path.exists():
        return None
    return pd.read_parquet(path)


