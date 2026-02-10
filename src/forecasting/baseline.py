"""
Baseline forecasting methods.

Tujuan modul ini:
- Menyediakan baseline super sederhana sebagai patokan:
  - Naive:          y_hat(t) = y(t-1)
  - Seasonal naive: y_hat(t) = y(t-52)  (untuk data mingguan)

Implementasi ini tidak bergantung pada framework ML tertentu, hanya NumPy/Pandas.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def global_naive_forecast(h: int, mean: float, std: float) -> List[float]:
    """
    Global baseline jika tidak ada data historis per SKU-location.
    Mirip dengan asumsi demand konstan di sekitar mean dengan sedikit bump di akhir horizon.
    """
    base = np.full(h, mean)
    # simple seasonal-ish bump last steps of horizon
    if h >= 2:
        base[-2:] = mean + 0.5 * std
    return base.clip(min=0).astype(float).tolist()


def naive_and_seasonal_from_history(
    sub: pd.DataFrame,
    h: int,
    mean: float,
    std: float,
) -> Tuple[List[float], Optional[List[float]]]:
    """
    Baseline per SKU-location berbasis histori:

    Parameters
    ----------
    sub : pd.DataFrame
        Riwayat untuk satu kombinasi store_id–product_id dengan kolom:
        - 'units_sold'
        - opsional: 'lag_52' untuk seasonal naive mingguan.
    h : int
        Horizon forecast (jumlah langkah ke depan).
    mean, std : float
        Statistik global fallback bila histori tidak cukup.

    Returns
    -------
    naive : list[float]
        Naive baseline: ulangi nilai units_sold terakhir untuk seluruh horizon.
    seasonal : Optional[list[float]]
        Seasonal naive (berbasis lag_52) jika tersedia & tidak NaN, else None.
    """
    if sub is None or len(sub) == 0:
        from .baseline import global_naive_forecast  # local import to avoid cycles

        return global_naive_forecast(h, mean, std), None

    sub_sorted = sub.sort_values(["year", "week"])

    # Naive: last actual value
    last_units = float(sub_sorted["units_sold"].iloc[-1])
    naive = [float(max(0.0, last_units))] * h

    seasonal: Optional[List[float]] = None
    if "lag_52" in sub_sorted.columns:
        val = sub_sorted["lag_52"].iloc[-1]
        if not pd.isna(val):
            seasonal_val = float(val)
            seasonal = [float(max(0.0, seasonal_val))] * h

    return naive, seasonal


